"""Isolated current-PR ERalpha benchmark, reusing the campaign's measurement helpers."""
import json
import os
from pathlib import Path
import shutil
import sys

BENCH = Path('/work/greencenter/s191084/cimist_perf/benchmarks')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'pr-newton-performance'
REFERENCE = BENCH / 'results/configurable20261018_mainline_candidate_em8_pair128__er_dimer_full'
sys.path.insert(0, str(BENCH))
import worker
import retain_outputs
import exact_h5_slabs
import row_slab_repack
from validate_batch16 import check_inputs


def read(path):
    return json.loads(path.read_text())


def verify_source():
    expected = read(SOURCE / 'docs/performance/source-hashes.json')
    actual = {name: worker.digest(SOURCE/name) for name in expected}
    if actual != expected:
        raise ValueError('Current PR source differs from reviewed manifest')
    return actual


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires a SLURM allocation')
    helpers = ('worker.py', 'mglu_sparse_instrument.py', 'retain_outputs.py',
               'row_slab_repack.py', 'exact_h5_slabs.py', 'exact_h5.py',
               'validate_batch16.py', 'matrix.json', 'output_experiment.py')
    pins = {name: worker.digest(BENCH/name) for name in helpers}
    if pins != read(HERE/'helper-hashes.json'):
        raise ValueError('Benchmark helpers changed since preparation')
    verify_source()
    old_prepare, old_run, old_validate = worker.prepare, worker.measured_run, worker.validate_output
    reports = []

    def prepare(case, scratch, report):
        reports.append(report)
        (HERE/'run-location.json').write_text(json.dumps({'report': str(report), 'job': os.environ['SLURM_JOB_ID']}))
        if case['id'] != 'er_dimer_full':
            raise ValueError('Wrong benchmark case')
        shutil.copytree(SOURCE/'src', report/'executed-source/src')
        for name in helpers:
            shutil.copy2(BENCH/name, report/('executed__'+name))
        shutil.copy2(Path(__file__), report/'run_er_full.py')
        sample = old_prepare(case, scratch, report)
        data = read(report/'input.json')
        if data['selected_global_indices'] != list(range(11082)) or data['protein_residues'] != 500:
            raise ValueError('Full ERalpha input shape mismatch')
        (report/'input-equivalence.json').write_text(json.dumps(check_inputs(read(REFERENCE/'input.json'), data), indent=2))
        host = read(report/'host.json')
        host['instrument_sha256'] = pins['mglu_sparse_instrument.py']
        host['source_commit'] = '588f0c7'
        host['reference_report'] = str(REFERENCE)
        (report/'host.json').write_text(json.dumps(host, indent=2))
        return sample

    def measured(command, report, output):
        verify_source()
        if command[1] != str(BENCH/'instrument.py'):
            raise ValueError('Unexpected instrumentation route')
        command = list(command)
        command[1] = str(BENCH/'mglu_sparse_instrument.py')
        command += ['--solver', 'newton', '--PAIR_BATCH_SIZE', '128',
                    '--probability_storage', 'sparse', '--hdf5_compression', 'none']
        (report/'command.json').write_text(json.dumps(command, indent=2))
        result = old_run(command, report, output)
        verify_source()
        return result

    def validate(path, frames):
        import h5py
        result = old_validate(path, frames)
        if frames != 11082 or result['failed_m_steps']:
            raise ValueError('Incorrect frames or failed Newton M-steps')
        with h5py.File(path) as f:
            if (f['VMM/mu'].shape[:2] != (500,100)
                    or f['states/states_traj'].shape != (500,11082)
                    or f['tree'].attrs['prior'] != 'haldane'
                    or f['tree'].attrs['entropy_estimator'] != 'bayes'):
                raise ValueError('Model configuration mismatch')
        result['scope'] = 'Execution validation only; not BFGS equivalence or global optimality'
        return result

    worker.prepare, worker.measured_run, worker.validate_output = prepare, measured, validate
    worker.compare_h5 = retain_outputs.compare_h5 = exact_h5_slabs.compare_h5_slabs
    retain_outputs.repack = row_slab_repack.repack
    cases = read(BENCH/'matrix.json')['cases']
    index = next(i for i,c in enumerate(cases) if c['id']=='er_dimer_full')
    sys.argv = [str(BENCH/'worker.py'), str(index), '--source-directory', str(SOURCE),
                '--result-tag', 'prnewton588f0c7_review20261020']
    worker.main()
    for report in reports:
        result = read(report/'SUCCESS.json')
        result.update(scope='Current Newton PR full ERalpha execution and checked lossless retention; scientific comparison is separate',
                      source_commit='588f0c7', hdf5_compression='none', helper_sha256=pins)
        (report/'SUCCESS.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
