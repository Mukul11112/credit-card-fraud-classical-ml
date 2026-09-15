from locust import HttpUser, task, between
import os


class APIUser(HttpUser):

    wait_time = between(1, 2)

    @task
    def predict_image(self):
        image_path = os.path.join(
            os.path.dirname(__file__),
            "test.jpg"
        )

        with open(image_path, "rb") as image:

            files = {
                "file": (
                    "test.jpg",
                    image,
                    "image/jpeg"
                )
            }

            self.client.post(
                "/predict",
                files=files,
                name="/predict"
            )

    @task
    def health_check(self):
        self.client.get(
            "/health",
            name="/health"
        )