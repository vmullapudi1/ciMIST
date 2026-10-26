"""Missing CLI paths fail before the pipeline creates outputs or loads inputs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

from cimist import fit


class CLIRequiredInputs(unittest.TestCase):
    def test_each_missing_path_stops_before_pipeline_activity(self):
        paths = [("-t", "--trajectory", "trajectory.xtc"),
                 ("-s", "--topology", "topology.pdb"),
                 ("-o", "--output_prefix", "run")]
        for missing in ((0,), (1,), (2,), (0, 1, 2)):
            argv = [item for i, (flag, _, value) in enumerate(paths)
                    if i not in missing for item in (flag, value)]
            with self.subTest(missing=missing), \
                    patch("sys.argv", ["ci-mist", *argv]), \
                    patch.object(fit.os, "mkdir") as mkdir, \
                    patch.object(fit, "load_dataset") as load, \
                    redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    fit.main()
                self.assertEqual(error.exception.code, 2)
                message = stderr.getvalue().splitlines()[-1]
                self.assertIn("the following arguments are required:", message)
                for i in missing:
                    self.assertIn(f"{paths[i][0]}/{paths[i][1]}", message)
                mkdir.assert_not_called()
                load.assert_not_called()

    def test_help_needs_no_input_paths(self):
        with patch("sys.argv", ["ci-mist", "--help"]), \
                patch.object(fit.os, "mkdir") as mkdir, \
                patch.object(fit, "load_dataset") as load, \
                redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as error:
                fit.main()
            self.assertEqual(error.exception.code, 0)
            for flag in ("--trajectory", "--topology", "--output_prefix"):
                self.assertIn(flag, stdout.getvalue())
            mkdir.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
