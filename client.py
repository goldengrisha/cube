import argparse
import base64
import io
import logging
import pathlib

import numpy as np
from PIL import Image  # pytype: disable=import-error

from pytriton.client import ModelClient

logger = logging.getLogger("examples.huggingface_stable_diffusion.client")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results",
        help="Path to folder where images should be stored.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    )

    prompts = [
        "A photo of an astronaut riding a horse on mars",
        "An image of a squirrel in Picasso style",
        "A running dog in the fields of trees in Manga style",
    ]

    img_size = np.array([[512]])
    results_path = pathlib.Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    with ModelClient(
        args.url, "StableDiffusion_1_5", init_timeout_s=args.init_timeout_s
    ) as client:
        for req_idx in range(1, args.iterations + 1):
            logger.debug(f"Sending request ({req_idx}).")
            prompt_id = req_idx % len(prompts)
            prompt = prompts[prompt_id]
            prompt = np.array([[prompt]])
            prompt = np.char.encode(prompt, "utf-8")
            logger.info(f"Prompt ({req_idx}): {prompt}")
            logger.info(f"Image size ({req_idx}): {img_size}")
            result_dict = client.infer_batch(prompt=prompt, img_size=img_size)
            logger.debug(f"Result for for request ({req_idx}).")

            for idx, image in enumerate(result_dict["image"]):
                file_idx = req_idx + idx
                file_path = results_path / str(file_idx) / "image.jpeg"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                msg = base64.b64decode(image[0])
                buffer = io.BytesIO(msg)
                image = Image.open(buffer)
                with file_path.open("wb") as fp:
                    image.save(fp)
                logger.info(f"Image saved to {file_path}")


if __name__ == "__main__":
    main()
