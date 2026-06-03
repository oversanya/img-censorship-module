from __future__ import annotations

import unittest

from censor_guard.app import app, ui_dir


class AppFrontendTests(unittest.TestCase):
    def test_registers_frontend_routes(self) -> None:
        route_paths = {getattr(route, "path", None) for route in app.routes}

        self.assertIn("/", route_paths)
        self.assertIn("/ui", route_paths)
        self.assertTrue((ui_dir / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
