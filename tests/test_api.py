import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed")
class ApiTest(unittest.TestCase):
    def test_health(self):
        from fastapi.testclient import TestClient
        from img_censor.api import app

        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_prompt_only_censor_allows_safe_prompt(self):
        from fastapi.testclient import TestClient
        from img_censor import api

        api.USE_MOCK = True

        response = TestClient(api.app).post(
            "/v1/censor",
            data={"prompt": "Сгенерируй фото машины"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "allow")


if __name__ == "__main__":
    unittest.main()
