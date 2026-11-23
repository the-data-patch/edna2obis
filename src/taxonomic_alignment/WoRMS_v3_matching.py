import pyworms
from pyworms import utils as pyworms_utils
import pandas as pd
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocess as mp
import time
from functools import partial
import logging
import re
import sys
import os
import pickle

# Set up logging to provide clear progress updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Standard Darwin Core ranks used for structuring the output
DWC_RANKS_STD = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

# Ordered WoRMS lineage fields used to build Darwin Core's `higherClassification`.
# Skip `superdomain` because WoRMS commonly returns the unhelpful root `Biota`.
WORMS_HIGHER_CLASSIFICATION_RANKS = [
    'kingdom',
    'subkingdom',
    'infrakingdom',
    'phylum',
    'subphylum',
    'infraphylum',
    'superclass',
    'megaclass',
    'class',
    'subclass',
    'infraclass',
    'superorder',
    'order',
    'suborder',
    'infraorder',
    'superfamily',
    'family',
    'subfamily',
    'tribe',
    'subtribe',
    'genus',
    'subgenus',
    'section',
    'subsection',
    'series',
    'subseries',
    'species',
    'subspecies'
]

# Parallel WoRMS classification fetches for higherClassification (I/O bound; cap limits API load).
_HIGHER_CLASSIFICATION_MAX_WORKERS = 6

# higherClassification lookups go straight to the WoRMS REST API (not through pyworms), so that
# an unresponsive WoRMS fails within minutes instead of holding each request for up to ~56 minutes.
_WORMS_REST_URL = 'https://www.marinespecies.org/rest/'
_WORMS_HC_TIMEOUT = (10, 30)           # seconds: (connect, read between bytes)
_WORMS_HC_RETRIES = 3                  # retries after the first attempt
_WORMS_HC_BACKOFF_SECONDS = 2          # wait before retry n is 2, 4, 8 s
_WORMS_HC_MAX_CONSECUTIVE_FAILURES = 10  # stop all lookups after this many failures in a row
_HIGHER_CLASSIFICATION_SAVE_EVERY = 50   # write the cache after this many new results
_HIGHER_CLASSIFICATION_CACHE_FILENAME = 'worms_higherClassification_cache.pkl'

_thread_local = threading.local()


def _format_environment(record):
    """
    Build a semicolon-separated environment label from WoRMS habitat flags.
    Leave blank when WoRMS does not provide any environment information.
    """
    if not isinstance(record, dict):
        return None

    environment_map = [
        ('isMarine', 'marine'),
        ('isBrackish', 'brackish'),
        ('isFreshwater', 'freshwater'),
        ('isTerrestrial', 'terrestrial')
    ]
    environments = []

    for field_name, label in environment_map:
        value = record.get(field_name)
        if value in (1, True, '1', 'true', 'True'):
            environments.append(label)

    return ';'.join(environments) if environments else None


def _format_higher_classification(classification):
    """
    Build a Darwin Core higherClassification string from a flattened WoRMS classification.
    WoRMS returns a flat rank dictionary here, not a nested `child` tree.
    Use ' | ' as the separator and exclude the matched leaf taxon itself.
    """
    if not isinstance(classification, dict) or not classification:
        return None

    lineage = []
    seen_names = set()
    for rank_name in WORMS_HIGHER_CLASSIFICATION_RANKS:
        value = classification.get(rank_name)
        if value is None or pd.isna(value):
            continue

        cleaned_value = str(value).strip()
        if not cleaned_value:
            continue

        normalized_value = cleaned_value.lower()
        if normalized_value in seen_names:
            continue

        seen_names.add(normalized_value)
        lineage.append(cleaned_value)

    if len(lineage) <= 1:
        return None

    higher_taxa = lineage[:-1]
    return ' | '.join(higher_taxa) if higher_taxa else None


class WormsRequestError(Exception):
    """Raised when WoRMS could not be reached or did not answer after all retries."""


def _get_worms_session():
    """
    Return a requests.Session for the current thread (Session is not guaranteed thread-safe).
    """
    session = getattr(_thread_local, 'worms_session', None)
    if session is None:
        session = requests.Session()
        _thread_local.worms_session = session
    return session


def _fetch_worms_classification(aphia_id):
    """
    Fetch the WoRMS classification for an AphiaID and flatten it to a rank dictionary.

    Unlike pyworms (300 s timeout, up to 10 retries), this gives up on a request that
    stalls for _WORMS_HC_TIMEOUT and retries only _WORMS_HC_RETRIES times, so an
    unresponsive WoRMS cannot hold a worker for close to an hour.

    Returns the flattened classification, or None when WoRMS has no classification
    for this AphiaID. Raises WormsRequestError when no answer could be obtained.
    """
    url = f'{_WORMS_REST_URL}AphiaClassificationByAphiaID/{int(aphia_id)}'
    last_error = None

    for attempt in range(_WORMS_HC_RETRIES + 1):
        if attempt:
            time.sleep(_WORMS_HC_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        try:
            response = _get_worms_session().get(url, timeout=_WORMS_HC_TIMEOUT)
        except requests.RequestException as e:
            last_error = f'{type(e).__name__}: {e}'
            continue

        # Same meaning as in pyworms: 204 and 400 are "nothing found", not failures.
        if response.status_code in (204, 400):
            return None
        if response.status_code == 200:
            try:
                return pyworms_utils.flatten(response.json())
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                raise WormsRequestError(f'Unexpected response for AphiaID {aphia_id}: {e}') from e

        last_error = f'HTTP {response.status_code}'
        # Only server-side errors and rate limiting are worth retrying.
        if response.status_code != 429 and response.status_code < 500:
            break

    raise WormsRequestError(f'AphiaID {aphia_id}: {last_error}')


def _extract_aphia_id_from_lsid(scientific_name_id):
    """
    Extract the numeric AphiaID from a WoRMS LSID string.
    """
    if scientific_name_id is None or pd.isna(scientific_name_id):
        return None

    match = re.search(r'(\d+)$', str(scientific_name_id).strip())
    if not match:
        return None

    try:
        return int(match.group(1))
    except Exception:
        return None


def _get_higher_classification_worker(scientific_name_id):
    """
    Fetch higherClassification for one scientificNameID / AphiaID.

    Returns (scientific_name_id, status, value), where status is:
      'ok'     - value holds the higherClassification string
      'empty'  - WoRMS answered but has no usable lineage for this taxon
      'failed' - WoRMS could not be reached; value holds the error message
    """
    aphia_id = _extract_aphia_id_from_lsid(scientific_name_id)
    if aphia_id is None:
        return scientific_name_id, 'empty', None

    try:
        classification = _fetch_worms_classification(aphia_id)
    except WormsRequestError as e:
        return scientific_name_id, 'failed', str(e)

    value = _format_higher_classification(classification)
    if value is None:
        return scientific_name_id, 'empty', None
    return scientific_name_id, 'ok', value


def _higher_classification_cache_path(cache_dir):
    return os.path.join(cache_dir, _HIGHER_CLASSIFICATION_CACHE_FILENAME)


def _load_higher_classification_cache(cache_dir):
    """
    Load cached higherClassification lookups from disk if available.
    """
    if not cache_dir:
        return {}

    cache_path = _higher_classification_cache_path(cache_dir)
    if not os.path.exists(cache_path):
        return {}

    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            return {}
        # Ignore empty/failed cached values so a bad run does not poison future runs.
        return {
            str(k).strip(): str(v).strip()
            for k, v in cache.items()
            if k is not None and not pd.isna(k) and v is not None and not pd.isna(v) and str(v).strip()
        }
    except Exception:
        return {}


def _save_higher_classification_cache(cache_dir, cache):
    """
    Persist higherClassification lookups so repeated runs do not re-fetch them.
    Writes to a temporary file and renames it, so an interrupted write cannot corrupt the cache.
    """
    if not cache_dir or not isinstance(cache, dict):
        return

    cache_path = _higher_classification_cache_path(cache_dir)
    tmp_path = cache_path + '.tmp'
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_to_save = {
            str(k).strip(): str(v).strip()
            for k, v in cache.items()
            if k is not None and not pd.isna(k) and v is not None and not pd.isna(v) and str(v).strip()
        }
        with open(tmp_path, 'wb') as f:
            pickle.dump(cache_to_save, f)
        os.replace(tmp_path, cache_path)
    except Exception as e:
        logging.warning(f'Could not save higherClassification cache to {cache_path}: {e}')


def _fetch_higher_classifications(ids_to_fetch, cache, cache_dir, n_workers):
    """
    Fetch higherClassification for ids_to_fetch, adding results to cache as they arrive.

    - Saves the cache every _HIGHER_CLASSIFICATION_SAVE_EVERY new results and again on exit,
      including on Ctrl-C or an error, so an interrupted run keeps its progress.
    - Stops sending requests after _WORMS_HC_MAX_CONSECUTIVE_FAILURES failures in a row
      (WoRMS is most likely down), instead of working through every remaining taxon.

    Returns a dict of counts: ok, empty, failed, skipped.
    """
    counts = {'ok': 0, 'empty': 0, 'failed': 0, 'skipped': 0}
    total = len(ids_to_fetch)
    unsaved = 0
    consecutive_failures = 0
    last_error = None
    done = 0
    progress_every = max(1, min(100, total // 10))

    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures = [executor.submit(_get_higher_classification_worker, sid) for sid in ids_to_fetch]
    try:
        for future in as_completed(futures):
            scientific_name_id, status, value = future.result()
            counts[status] += 1
            done += 1

            if status == 'failed':
                consecutive_failures += 1
                last_error = value
            else:
                consecutive_failures = 0

            if status == 'ok':
                cache[scientific_name_id] = value
                unsaved += 1
                if unsaved >= _HIGHER_CLASSIFICATION_SAVE_EVERY:
                    _save_higher_classification_cache(cache_dir, cache)
                    unsaved = 0

            if done % progress_every == 0 or done == total:
                logging.info(f'  ... higherClassification progress: {done}/{total} '
                             f'({counts["failed"]} failed).')

            if consecutive_failures >= _WORMS_HC_MAX_CONSECUTIVE_FAILURES:
                logging.warning(
                    f'WoRMS appears to be unavailable: {consecutive_failures} higherClassification '
                    f'lookups failed in a row (last error: {last_error}). Stopping lookups.'
                )
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        counts['skipped'] = total - done
        if unsaved:
            _save_higher_classification_cache(cache_dir, cache)

    return counts


def _enrich_higher_classification_dataframes(source_df, target_dataframes, cache_dir='.', n_proc=1,
                                            legacy_cache_dir=None):
    """
    Populate higherClassification once per unique scientificNameID across the target outputs.
    This keeps the final occurrence core and taxa mapping info file aligned.

    cache_dir holds the shared lookup cache (outputs/ by default, so every run reuses it).
    legacy_cache_dir is an older per-run folder whose cache is merged in if present.
    """
    valid_targets = [
        df for df in target_dataframes
        if df is not None and not df.empty and 'scientificNameID' in df.columns
    ]
    if not valid_targets:
        return

    unique_scientific_name_ids = []
    seen_ids = set()
    dataframes_to_scan = []
    if source_df is not None and not source_df.empty and 'scientificNameID' in source_df.columns:
        dataframes_to_scan.append(source_df)
    dataframes_to_scan.extend(valid_targets)

    for df in dataframes_to_scan:
        for scientific_name_id in df['scientificNameID'].dropna().astype(str).str.strip().unique().tolist():
            aphia_id = _extract_aphia_id_from_lsid(scientific_name_id)
            if aphia_id is None or aphia_id == 12 or scientific_name_id in seen_ids:
                continue
            seen_ids.add(scientific_name_id)
            unique_scientific_name_ids.append(scientific_name_id)

    if not unique_scientific_name_ids:
        return

    cache = {}
    if legacy_cache_dir and os.path.abspath(legacy_cache_dir) != os.path.abspath(cache_dir or '.'):
        cache.update(_load_higher_classification_cache(legacy_cache_dir))
    cache.update(_load_higher_classification_cache(cache_dir))

    ids_to_fetch = [scientific_name_id for scientific_name_id in unique_scientific_name_ids if scientific_name_id not in cache]
    logging.info(
        f'higherClassification: {len(unique_scientific_name_ids) - len(ids_to_fetch)} of '
        f'{len(unique_scientific_name_ids)} taxa found in cache '
        f'({_higher_classification_cache_path(cache_dir)}).'
    )

    if ids_to_fetch:
        fetch_workers = min(max(int(n_proc or 1), 1), _HIGHER_CLASSIFICATION_MAX_WORKERS, len(ids_to_fetch))
        logging.info(f'Fetching higherClassification for {len(ids_to_fetch)} uncached WoRMS taxa '
                     f'with {fetch_workers} parallel requests.')
        counts = _fetch_higher_classifications(ids_to_fetch, cache, cache_dir, fetch_workers)

        logging.info(
            f'higherClassification lookups: {counts["ok"]} fetched, '
            f'{counts["empty"]} without a WoRMS lineage, {counts["failed"]} failed, '
            f'{counts["skipped"]} not attempted.'
        )
        missing = counts['failed'] + counts['skipped']
        if missing:
            logging.warning(
                f'{missing} taxa have no higherClassification because WoRMS could not be reached. '
                f'Rerun later to fill them in; the {len(cache)} lookups already done are cached in '
                f'{_higher_classification_cache_path(cache_dir)}.'
            )

    for df in valid_targets:
        if 'higherClassification' not in df.columns:
            df['higherClassification'] = pd.NA

        mapped_values = df['scientificNameID'].astype('string').map(cache)
        fill_mask = df['higherClassification'].isna() | (df['higherClassification'].astype('string').str.strip() == '')
        df.loc[fill_mask, 'higherClassification'] = mapped_values[fill_mask]


def _build_worms_result_record(record_to_use, api_source_for_record, replaced_unaccepted=False, unaccepted_match=False):
    """
    Format a WoRMS record into the edna2obis result structure.
    unaccepted_match: True when this row is the unaccepted (queried) taxon shown for transparency when WoRMS resolved to a different accepted name.
    """
    result = {
        'scientificName': record_to_use.get('scientificname'),
        'scientificNameID': record_to_use.get('lsid'),
        'taxonRank': record_to_use.get('rank'),
        'nameAccordingTo': api_source_for_record,
        'replaced_unaccepted': replaced_unaccepted,
        'unaccepted_match': unaccepted_match,
        'environment': _format_environment(record_to_use)
    }
    for rank_std in DWC_RANKS_STD:
        result[rank_std] = record_to_use.get(rank_std.lower())
    result['higherClassification'] = pd.NA

    return result

def _normalize_taxon_token(s):
    """
    Normalize taxonomy tokens for loose equality checks.
    Keep it conservative: we only want to match obvious identical terms.
    """
    if s is None or pd.isna(s):
        return None
    s = str(s).strip().lower()
    if not s:
        return None
    # Normalize common separators; keep letters/spaces to avoid over-matching.
    s = s.replace('_', ' ').replace('-', ' ').replace('/', ' ')
    if '  ' in s:
        s = re.sub(r'\s+', ' ', s)
    return s.strip() or None


def _compute_lineage_consistency(parsed_names, candidate_match):
    """
    Returns:
      (score, matched_count, total_count)

    Score in [0, 1]: fraction of verbatim tokens that appear somewhere in the
    candidate's returned classification (including scientificName).

    This is intentionally simple and rank-agnostic because verbatim strings
    often include non-standard ranks (e.g., tribes) and "Eukaryota" as a superkingdom.
    """
    if not parsed_names or not isinstance(candidate_match, dict):
        return 0.0, 0, 0

    # Treat Eukaryota as an optional superkingdom token (WoRMS returns Animalia/Plantae/etc as 'kingdom').
    ignore = {'eukaryota'}
    verbatim_tokens = []
    for t in parsed_names:
        nt = _normalize_taxon_token(t)
        if not nt or nt in ignore:
            continue
        verbatim_tokens.append(nt)
    verbatim_set = set(verbatim_tokens)
    if not verbatim_set:
        return 0.0, 0, 0

    candidate_vals = set()
    sci = _normalize_taxon_token(candidate_match.get('scientificName'))
    if sci:
        candidate_vals.add(sci)
    for rank in DWC_RANKS_STD:
        v = _normalize_taxon_token(candidate_match.get(rank))
        if v:
            candidate_vals.add(v)

    if not candidate_vals:
        return 0.0, 0, len(verbatim_set)

    matched = verbatim_set.intersection(candidate_vals)
    total = len(verbatim_set)
    matched_count = len(matched)
    return matched_count / max(1, total), matched_count, total


def _candidate_completeness(candidate_match):
    """
    Tie-breaker that prefers records with more populated classification fields.
    This avoids biasing toward 'species' purely because it's a more specific rank.
    """
    if not isinstance(candidate_match, dict):
        return 0
    n = 0
    if _normalize_taxon_token(candidate_match.get('scientificName')):
        n += 1
    for rank in DWC_RANKS_STD:
        if _normalize_taxon_token(candidate_match.get(rank)):
            n += 1
    return n


def _get_mp_context():
    """
    On macOS, forking can crash with Objective-C runtime errors like:
    'may have been in progress in another thread when fork() was called'.
    Using 'spawn' avoids fork() and is the safest cross-platform default.
    """
    if sys.platform == 'darwin':
        return mp.get_context('spawn')
    return mp.get_context()

def _safe_str_or_na(x):
    """Coerce to Python str or pd.NA. Avoids Mac/Linux pandas dtype 'str' error from numpy scalars/bytes."""
    if pd.isna(x) or x is None:
        return pd.NA
    return str(x).strip() if isinstance(x, str) else str(x)


def parse_semicolon_taxonomy(tax_string):
    """
    Helper function to parse and clean a semicolon-separated taxonomy string.
    """
    if pd.isna(tax_string) or not str(tax_string).strip():
        return []
    
    # Fast string cleaning with chained replacements
    cleaned_string = str(tax_string).replace('_', ' ').replace('-', ' ').replace('/', ' ')
    
    # Split and clean in one pass
    cleaned_names = []
    for name in cleaned_string.split(';'):
        name = name.strip()
        if not name or name.lower() == 'unassigned':
            continue
            
        # Fast cleaning without regex
        name = name.replace(' sp.', '').replace(' spp.', '')
        
        # Remove numbers - only use regex once per name if needed
        if any(char.isdigit() for char in name):
            name = re.sub(r'\d+', '', name)
        
        # Collapse multiple spaces - only if needed
        if '  ' in name:
            name = re.sub(r'\s+', ' ', name)
        
        name = name.strip()
        if name and len(name) > 1:
            cleaned_names.append(name)
    
    return cleaned_names

def get_worms_classification_by_id_worker(aphia_id_to_check, api_source_for_record='WoRMS'):
    """Fetches and formats a full WoRMS record using a direct AphiaID.
    Used by local database pre-matching. Example uses Silva PR2 database"""
    try:
        record = pyworms.aphiaRecordByAphiaID(aphia_id_to_check)
        
        if record and isinstance(record, dict) and record.get('status') == 'accepted':
            result = _build_worms_result_record(
                record_to_use=record,
                api_source_for_record=api_source_for_record,
                replaced_unaccepted=False,
                unaccepted_match=False
            )
            result['match_type_debug'] = f'Success_AphiaID_{aphia_id_to_check}'
            return aphia_id_to_check, result
    except Exception:
        pass
    
    return aphia_id_to_check, {'match_type_debug': f'Failure_AphiaID_{aphia_id_to_check}', 'replaced_unaccepted': False, 'unaccepted_match': False}

def get_worms_batch_worker(batch_info):
    """
    Worker function for parallel batch processing - with timeout safety.
    Now collects ALL accepted matches to handle ambiguous cases.
    """
    batch_num, chunk, marine_only = batch_info
    batch_results = {}
    
    try:
        batch_results_raw = pyworms.aphiaRecordsByMatchNames(chunk, marine_only=marine_only)

        for j, name_list in enumerate(batch_results_raw):
            if not name_list:
                continue

            accepted_by_lsid = {}
            unaccepted_rows_for_term = []

            for match in name_list:
                if not match:
                    continue

                status = match.get('status')
                record_to_use = None
                name_changed = False

                # Case 1: already an accepted record
                if status == 'accepted':
                    record_to_use = match

                # Case 2: unaccepted record with a valid (accepted) name behind it.
                # For example: Synagrops spinosus -> Parascombrops spinosus.
                elif status == 'unaccepted':
                    valid_aphia = (
                        match.get('valid_AphiaID')
                        or match.get('valid_aphia_id')
                        or match.get('valid_aphiaID')
                    )
                    if valid_aphia:
                        try:
                            resolved = pyworms.aphiaRecordByAphiaID(valid_aphia)
                            if resolved and resolved.get('status') == 'accepted':
                                record_to_use = resolved
                                name_changed = True
                        except Exception:
                            record_to_use = None

                # Ignore any other statuses
                if not record_to_use:
                    continue

                sci_id = record_to_use.get('lsid')
                if not sci_id:
                    continue

                # Only after we have a usable accepted record (with lsid): add the unaccepted row for transparency.
                if name_changed and status == 'unaccepted':
                    unaccepted_rows_for_term.append(_build_worms_result_record(
                        record_to_use=match,
                        api_source_for_record='WoRMS',
                        replaced_unaccepted=False,
                        unaccepted_match=True
                    ))

                existing = accepted_by_lsid.get(sci_id)
                if existing is None:
                    res = _build_worms_result_record(
                        record_to_use=record_to_use,
                        api_source_for_record='WoRMS',
                        replaced_unaccepted=name_changed,
                        unaccepted_match=False
                    )
                    accepted_by_lsid[sci_id] = res
                else:
                    if name_changed and not existing.get('replaced_unaccepted'):
                        existing['replaced_unaccepted'] = True

            accepted_matches = list(accepted_by_lsid.values()) + unaccepted_rows_for_term

            if accepted_matches:
                batch_results[chunk[j]] = accepted_matches
            
        return batch_num, batch_results

    except Exception:
        return batch_num, {}

def get_worms_match_for_dataframe(occurrence_df, params_dict, n_proc=0):
    """
    Adds WoRMS taxonomic information and generates a detailed info file for ambiguous matches.
    
    Returns:
        dict: A dictionary containing two DataFrames:
              - 'main_df': The occurrence DataFrame with a single best match (maintains original behavior).
              - 'info_df': A detailed DataFrame with all possible matches for ambiguous cases.
    """
    api_source = params_dict.get('taxonomic_api_source', 'WoRMS')
    assays_to_skip_species = params_dict.get('assays_to_skip_species_match', [])
    
    # Handle case where parameter is None
    if assays_to_skip_species is None:
        assays_to_skip_species = []
        
    pr2_dict = params_dict.get('pr2_worms_dict', {})
    assay_rank_info = params_dict.get('assay_rank_info', {})

    worms_min_ranks_matched = params_dict.get('worms_min_ranks_matched', 1)
    try:
        worms_min_ranks_matched = int(worms_min_ranks_matched)
    except Exception:
        worms_min_ranks_matched = 1
    if worms_min_ranks_matched < 1:
        worms_min_ranks_matched = 1

    worms_max_walkup_steps = params_dict.get('worms_max_walkup_steps', None)
    if worms_max_walkup_steps is not None:
        try:
            worms_max_walkup_steps = int(worms_max_walkup_steps)
        except Exception:
            worms_max_walkup_steps = None
        if worms_max_walkup_steps is not None and worms_max_walkup_steps < 0:
            worms_max_walkup_steps = 0

    worms_return_all_matches = params_dict.get('worms_return_all_matches', False)
    if isinstance(worms_return_all_matches, str):
        worms_return_all_matches = worms_return_all_matches.strip().lower() in {'1', 'true', 'yes', 'y'}
    else:
        worms_return_all_matches = bool(worms_return_all_matches)
    marine_only = not worms_return_all_matches

    worms_return_higher_classification = params_dict.get('worms_return_higher_classification', False)
    if isinstance(worms_return_higher_classification, str):
        worms_return_higher_classification = worms_return_higher_classification.strip().lower() in {'1', 'true', 'yes', 'y'}
    else:
        worms_return_higher_classification = bool(worms_return_higher_classification)

    worms_consider_unaccepted_for_selection = params_dict.get('worms_consider_unaccepted_for_selection', False)
    if isinstance(worms_consider_unaccepted_for_selection, str):
        worms_consider_unaccepted_for_selection = worms_consider_unaccepted_for_selection.strip().lower() in {'1', 'true', 'yes', 'y'}
    else:
        worms_consider_unaccepted_for_selection = bool(worms_consider_unaccepted_for_selection)

    walkup_stats = {
        'min_ranks_matched': worms_min_ranks_matched,
        'max_walkup_steps': worms_max_walkup_steps,
        'marine_only': marine_only,
        'return_higher_classification': worms_return_higher_classification,
        'considered_terms': 0,
        'rejected_terms': 0,
        'accepted_matches': 0,
        'walked_up_matches': 0,
        'accepted_at_lowest': 0,
        'max_step_used': 0,
        'no_acceptable_term_found': 0
    }

    if n_proc == 0:
        n_proc = min(10, mp.cpu_count())
    else:
        n_proc = min(int(n_proc), 10)
    params_dict['worms_n_proc_effective'] = n_proc
    logging.info(f"Using {n_proc} processes for WoRMS matching (capped at 10)")
    if n_proc >= 8:
        logging.warning("WoRMS: %s parallel workers — may hit API rate limits or errors", n_proc)

    df_to_process = occurrence_df.copy()
    # Safe string conversion (avoids Mac/pandas dtype 'str' error)
    df_to_process['verbatimIdentification'] = df_to_process['verbatimIdentification'].apply(
        lambda x: '' if pd.isna(x) else str(x).strip()
    )
    if 'assay_name' in df_to_process.columns:
        df_to_process['assay_name'] = df_to_process['assay_name'].apply(
            lambda x: '' if pd.isna(x) else str(x).strip()
        )

    empty_verbatim_mask = (df_to_process['verbatimIdentification'].str.strip() == '') | \
                          (df_to_process['verbatimIdentification'].str.strip() == '') | \
                          (df_to_process['verbatimIdentification'].str.strip().str.lower() == 'unassigned')
    
    # IMPORTANT: This column must be able to hold tuples (verbatimIdentification, assay_name).
    # On some Mac/Linux pandas installs, assigning '' can create a strict string dtype ('str'),
    # which then rejects tuples with: "Invalid value for dtype 'str'".
    df_to_process['_map_key'] = pd.Series([None] * len(df_to_process), index=df_to_process.index, dtype='object')
    df_to_process.loc[empty_verbatim_mask, '_map_key'] = 'IS_TRULY_EMPTY'
    
    non_empty_mask = ~empty_verbatim_mask
    
    keys_to_assign = pd.Series(
        list(zip(
            df_to_process.loc[non_empty_mask, 'verbatimIdentification'],
            df_to_process.loc[non_empty_mask, 'assay_name']
        )),
        index=df_to_process[non_empty_mask].index
    )
    df_to_process.loc[non_empty_mask, '_map_key'] = keys_to_assign
    
    unique_tuples_to_process = list(df_to_process.loc[non_empty_mask, '_map_key'].drop_duplicates())
    logging.info(f"Found {len(unique_tuples_to_process)} unique, non-empty combinations to process.")

    results_cache = {}
    info_records = []  # NEW: To build the detailed info DataFrame
    unmatched_tuples = []
    
    cases_to_handle = []
    for verbatim_str, assay_name in unique_tuples_to_process:
        cleaned_verbatim = str(verbatim_str).strip().rstrip(';').strip()
        
        is_unassigned = (not cleaned_verbatim or 
                         cleaned_verbatim.lower() in ['unassigned', 'nan', 'none', ''] or
                         pd.isna(verbatim_str))
        
        is_simple_kingdom = cleaned_verbatim.lower() in ['eukaryota']

        if is_unassigned or is_simple_kingdom:
            match_type = 'incertae_sedis_unassigned' if is_unassigned else f'incertae_sedis_simple_case_{cleaned_verbatim}'
            incertae_sedis_record = {
                'scientificName': 'incertae sedis',
                'scientificNameID': 'urn:lsid:marinespecies.org:taxname:12', 'taxonRank': None,
                'nameAccordingTo': api_source, 'match_type_debug': match_type,
                'cleanedTaxonomy': cleaned_verbatim, 'replaced_unaccepted': False, 'unaccepted_match': False,
                'environment': pd.NA,
                'higherClassification': pd.NA,
                'assignment_score': pd.NA,
                'ranks_matched': pd.NA,
                'ranks_provided': pd.NA
            }
            for col in DWC_RANKS_STD:
                incertae_sedis_record[col] = None
            
            results_cache[(verbatim_str, assay_name)] = incertae_sedis_record
            info_records.append({'verbatimIdentification': verbatim_str, **incertae_sedis_record, 'ambiguous': False})
            cases_to_handle.append((verbatim_str, assay_name))
    
    unique_tuples_to_process = [t for t in unique_tuples_to_process if t not in cases_to_handle]
    if cases_to_handle:
        logging.info(f"Assigned {len(cases_to_handle)} cases (unassigned/empty/simple) to 'incertae sedis'.")

    # --- Stage 1: PR2 AphiaID Pre-matching ---
    if pr2_dict:
        logging.info("Starting Stage 1: Parallel AphiaID pre-matching.")
        aphia_id_map = {}
        stage1_unmatched = []
        
        for verbatim_str, assay_name in unique_tuples_to_process:
            if assay_name in assays_to_skip_species:
                stage1_unmatched.append((verbatim_str, assay_name))
                continue
            
            parsed_names = parse_semicolon_taxonomy(verbatim_str)
            if parsed_names:
                species_name = parsed_names[-1]
                if species_name in pr2_dict:
                    aphia_id = pr2_dict[species_name]
                    if aphia_id not in aphia_id_map:
                        aphia_id_map[aphia_id] = []
                    aphia_id_map[aphia_id].append((verbatim_str, assay_name))
                    continue
            stage1_unmatched.append((verbatim_str, assay_name))
        
        unique_aphia_ids_to_fetch = list(aphia_id_map.keys())
        if unique_aphia_ids_to_fetch:
            ctx = _get_mp_context()
            with ctx.Pool(processes=n_proc) as pool:
                worker_func = partial(get_worms_classification_by_id_worker, api_source_for_record=api_source)
                parallel_results = pool.map(worker_func, unique_aphia_ids_to_fetch)

            for aphia_id, result in parallel_results:
                if 'scientificName' in result:
                    for combo in aphia_id_map.get(aphia_id, []):
                        verbatim_str, _ = combo
                        parsed_names = parse_semicolon_taxonomy(verbatim_str)
                        cleaned_taxonomy = ';'.join(parsed_names) if parsed_names else verbatim_str
                        consistency_score, matched_count, total_count = _compute_lineage_consistency(parsed_names, result)
                        result_with_cleaned = {
                            **result,
                            'cleanedTaxonomy': cleaned_taxonomy,
                            'assignment_score': consistency_score,
                            'ranks_matched': matched_count,
                            'ranks_provided': total_count
                        }
                        
                        results_cache[combo] = result_with_cleaned
                        info_records.append({'verbatimIdentification': verbatim_str, **result_with_cleaned, 'ambiguous': False})
                else:
                    stage1_unmatched.extend(aphia_id_map.get(aphia_id, []))
        
        unmatched_tuples = stage1_unmatched
        logging.info(f"Finished Stage 1. Matched {len(results_cache) - len(cases_to_handle)} taxa via AphiaID. Remaining: {len(unmatched_tuples)}.")
    else:
        unmatched_tuples = unique_tuples_to_process

    # --- Stage 2: SMART Parallel Batch Name Matching ---
    if unmatched_tuples:
        all_terms_to_match = list(set(term for vt, an in unmatched_tuples for term in parse_semicolon_taxonomy(vt)))
        logging.info(f"Starting Stage 2: Smart batch matching for {len(all_terms_to_match)} unique terms.")
        
        batch_lookup = {}
        if all_terms_to_match:
            chunk_size = 50
            batch_data = [((i // chunk_size) + 1, all_terms_to_match[i:i+chunk_size], marine_only) for i in range(0, len(all_terms_to_match), chunk_size)]
            total_batches = len(batch_data)
            
            logging.info(f"Processing {total_batches} batches with {n_proc} processes...")
            
            processed_batches = 0
            ctx = _get_mp_context()
            with ctx.Pool(processes=n_proc) as pool:
                # Use imap_unordered to get results as they complete, allowing for progress reporting
                for batch_num, batch_result in pool.imap_unordered(get_worms_batch_worker, batch_data):
                    batch_lookup.update(batch_result)
                    processed_batches += 1
                    logging.info(f"  ... Progress: {processed_batches}/{total_batches} batches completed (Batch ID: {batch_num}).")

            logging.info(f"Smart parallel processing complete! Found matches for {len(batch_lookup)} terms.")

            still_unmatched_batch = []
            for verbatim_str, assay_name in unmatched_tuples:
                parsed_names = parse_semicolon_taxonomy(verbatim_str)
                if not parsed_names:
                    still_unmatched_batch.append((verbatim_str, assay_name))
                    continue
                
                max_depth = assay_rank_info.get(assay_name, {}).get('max_depth', 99)
                is_full_len = len(parsed_names) >= max_depth
                
                if assay_name in assays_to_skip_species and is_full_len:
                    parsed_names = parsed_names[:-1]
                    if not parsed_names:
                        still_unmatched_batch.append((verbatim_str, assay_name))
                        continue
                
                cleaned_taxonomy = ';'.join(parsed_names) if parsed_names else verbatim_str
                match_found = False
                
                for walkup_step, term in enumerate(reversed(parsed_names)):
                    if worms_max_walkup_steps is not None and walkup_step > worms_max_walkup_steps:
                        break

                    if term in batch_lookup:
                        all_matches = batch_lookup[term]
                        is_ambiguous = len(all_matches) > 1
                        walkup_stats['considered_terms'] += 1
                        
                        # For selection (occurrence core), optionally exclude unaccepted-match rows so only accepted names can win.
                        candidates_for_selection = [m for m in all_matches if not m.get('unaccepted_match')] if not worms_consider_unaccepted_for_selection else all_matches
                        if not candidates_for_selection:
                            if worms_consider_unaccepted_for_selection:
                                candidates_for_selection = all_matches
                            else:
                                continue
                        
                        # Prefer exact scientificName match to the queried term when available,
                        # then choose by lineage consistency score (tie-breaker: more complete classification).
                        norm_term = _normalize_taxon_token(term)
                        exact_candidates = []
                        for m in candidates_for_selection:
                            if _normalize_taxon_token(m.get('scientificName')) == norm_term:
                                exact_candidates.append(m)
                        candidates = exact_candidates if exact_candidates else candidates_for_selection

                        scored_candidates = []
                        for m in candidates:
                            s, matched_count, total_count = _compute_lineage_consistency(parsed_names, m)
                            scored_candidates.append((s, _candidate_completeness(m), matched_count, total_count, m))
                        scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        best_score, _, best_matched_count, best_total_count, best_match = scored_candidates[0]

                        if best_matched_count < worms_min_ranks_matched:
                            walkup_stats['rejected_terms'] += 1
                            continue
                        
                        walkup_stats['accepted_matches'] += 1
                        if walkup_step == 0:
                            walkup_stats['accepted_at_lowest'] += 1
                        else:
                            walkup_stats['walked_up_matches'] += 1
                            if walkup_step > walkup_stats['max_step_used']:
                                walkup_stats['max_step_used'] = walkup_step

                        results_cache[(verbatim_str, assay_name)] = {
                            **best_match,
                            'nameAccordingTo': api_source,
                            'match_type_debug': f'Success_Batch_{term}',
                            'cleanedTaxonomy': cleaned_taxonomy,
                            'assignment_score': best_score,
                            'ranks_matched': best_matched_count,
                            'ranks_provided': best_total_count
                        }
                        
                        # For info DF, add all matches
                        for match in all_matches:
                            s, matched_count, total_count = _compute_lineage_consistency(parsed_names, match)
                            info_records.append({
                                'verbatimIdentification': verbatim_str, **match,
                                'nameAccordingTo': api_source, 'match_type_debug': f'Success_Batch_{term}',
                                'cleanedTaxonomy': cleaned_taxonomy,
                                'assignment_score': s,
                                'ranks_matched': matched_count,
                                'ranks_provided': total_count,
                                'ambiguous': is_ambiguous
                            })
                        
                        match_found = True
                        break
                
                if not match_found:
                    walkup_stats['no_acceptable_term_found'] += 1
                    still_unmatched_batch.append((verbatim_str, assay_name))
            
            unmatched_tuples = still_unmatched_batch
            logging.info(f"Stage 2 complete. Total matched: {len(results_cache)}. Remaining: {len(unmatched_tuples)}")

    params_dict['worms_walkup_stats'] = dict(walkup_stats)

    # --- Handle Final Unmatched ---
    if unmatched_tuples:
        for verbatim_str, assay_name in unmatched_tuples:
            parsed_names = parse_semicolon_taxonomy(verbatim_str)
            max_depth = assay_rank_info.get(assay_name, {}).get('max_depth', 99)
            is_full_len = len(parsed_names) >= max_depth
            
            if assay_name in assays_to_skip_species and is_full_len and parsed_names:
                parsed_names = parsed_names[:-1]
            
            cleaned_taxonomy = ';'.join(parsed_names) if parsed_names else verbatim_str
            
            no_match_record = {
                'scientificName': 'incertae sedis', 'scientificNameID': 'urn:lsid:marinespecies.org:taxname:12',
                'taxonRank': None, 'nameAccordingTo': api_source,
                'match_type_debug': 'Failed_All_Stages_NoMatch', 'cleanedTaxonomy': cleaned_taxonomy,
                'replaced_unaccepted': False, 'unaccepted_match': False,
                'environment': pd.NA,
                'higherClassification': pd.NA,
                'assignment_score': pd.NA,
                'ranks_matched': pd.NA,
                'ranks_provided': pd.NA
            }
            for col in DWC_RANKS_STD:
                no_match_record[col] = None

            results_cache[(verbatim_str, assay_name)] = no_match_record
            info_records.append({'verbatimIdentification': verbatim_str, **no_match_record, 'ambiguous': False})

    # --- Create record for initially empty inputs (for main DF) ---
    empty_fallback_record = { 
        'scientificName': 'incertae sedis', 'scientificNameID': 'urn:lsid:marinespecies.org:taxname:12',
        'taxonRank': None, 'nameAccordingTo': api_source,
        'match_type_debug': 'incertae_sedis_truly_empty_fallback', 'cleanedTaxonomy': '',
        'replaced_unaccepted': False, 'unaccepted_match': False,
        'environment': pd.NA,
        'higherClassification': pd.NA,
        'assignment_score': pd.NA,
        'ranks_matched': pd.NA,
        'ranks_provided': pd.NA
    }
    for col in DWC_RANKS_STD:
        empty_fallback_record[col] = None
    results_cache['IS_TRULY_EMPTY'] = empty_fallback_record

    # --- Apply results to main occurrence DataFrame ---
    results_df = pd.DataFrame()
    if results_cache:
        mapped_results = df_to_process['_map_key'].map(results_cache)
        results_df = pd.DataFrame(mapped_results.to_list(), index=df_to_process.index)
    info_df = pd.DataFrame(info_records)

    params_dict.pop('worms_higher_classification_seconds', None)
    if worms_return_higher_classification:
        hc_t0 = time.perf_counter()
        _enrich_higher_classification_dataframes(
            source_df=results_df,
            target_dataframes=[results_df, info_df],
            cache_dir=params_dict.get('worms_cache_dir') or params_dict.get('output_dir', '.'),
            n_proc=n_proc,
            legacy_cache_dir=params_dict.get('output_dir'),
        )
        params_dict['worms_higher_classification_seconds'] = time.perf_counter() - hc_t0

    if not results_df.empty:
        for col in results_df.columns:
            results_df[col] = results_df[col].apply(_safe_str_or_na)
        for col in results_df.columns:
            df_to_process[col] = results_df[col]

    df_to_process.drop(columns=['_map_key'], inplace=True, errors='ignore')
    
    # --- Create the detailed info DataFrame ---
    for col in info_df.columns:
        info_df[col] = info_df[col].apply(_safe_str_or_na)

    return {'main_df': df_to_process, 'info_df': info_df}

if __name__ == '__main__':
    pass