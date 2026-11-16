import ast
import unittest
from pathlib import Path


class BenchmarkRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(Path(__file__).with_name("modal_status_bench.py").read_text())
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def test_modal_decorator_registers_entrypoint_not_build_helper(self):
        self.assertEqual(self.functions["_build_gpu_binaries"].decorator_list, [])
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "function"
            for node in self.functions["prepare_cpu"].decorator_list
        ))

    def test_watchdog_uses_rank_zero_step_marker(self):
        function = self.functions["_run_multirank"]
        defaults = dict(zip(
            [arg.arg for arg in function.args.args][-len(function.args.defaults):],
            function.args.defaults,
        ))
        self.assertEqual(ast.literal_eval(defaults["init_marker"]), "[Rank 0] Step ")

    def test_completed_step_metrics_ignore_partial_and_other_rank_logs(self):
        function = self.functions["run_gpu_train_and_infer"]
        loop = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr == "splitlines"
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == "out_c"
        )
        scope = dict(
            out_c="\n".join([
                "[Rank 0] Step 3 start batch=2",
                "[Rank 1] Step 3 completed loss=99 recon=99 src_rms=99",
                "[Rank 0] Step 3 completed loss=0.125000 recon=0.000001 logdet=0.001 alpha=0.2 src_rms=0.250000 global_tokens=128 step_total_ms=23",
                "[Rank 0] Step invalid completed loss=0.1",
                "[Rank 0] Step 4 completed loss=0.062500 recon=0.000002 src_rms=0.500000",
            ]),
            timing_keys=[], timing_samples={}, throughput_samples=[],
            loss_curve=[], recon_curve=[], source_rms_curve=[], epoch_metrics=[],
        )
        module = ast.Module(body=[loop], type_ignores=[])
        exec(compile(module, "modal_status_bench.py", "exec"), scope)
        self.assertEqual(scope["loss_curve"], [(3, 0.125), (4, 0.0625)])
        self.assertEqual(scope["recon_curve"], [(3, 0.000001), (4, 0.000002)])
        self.assertEqual(scope["source_rms_curve"], [(3, 0.25), (4, 0.5)])
        self.assertEqual(scope["throughput_samples"], [(128, 23)])


if __name__ == "__main__":
    unittest.main()
