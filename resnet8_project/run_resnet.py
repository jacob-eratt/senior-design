"""Run the frozen FP32 or exported integer ResNet-8 on a 32x32 RGB image."""

import argparse
from pathlib import Path

from resnet8.common import ARTIFACTS, CLASSES, image_input, save_tensors, scores_text, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--integer", action="store_true", help="Use exported integer model")
    parser.add_argument("--export", type=Path, default=ARTIFACTS / "export")
    parser.add_argument("--output", type=Path, help="Tensor directory (default artifacts/inference/<image>/<mode>)")
    args = parser.parse_args()
    image = image_input(args.image)
    if args.integer:
        from resnet8.export import load_export
        model = load_export(args.export)
        tensors = model.run(image)
        scores = model.probabilities(tensors)[0]
    else:
        from resnet8.fp32 import Reference
        tensors = Reference().run(image)
        scores = tensors["probabilities"][0]
    mode = "integer" if args.integer else "fp32"
    directory = args.output or ARTIFACTS / "inference" / args.image.stem / mode
    save_tensors(directory, {"input_uint8": image, **tensors})
    write_json(directory / "prediction.json", {"mode": mode, "image": str(args.image),
               "class_scores": dict(zip(CLASSES, scores.tolist())), "prediction": CLASSES[int(scores.argmax())]})
    print(scores_text(scores))
    print(f"\nTensors: {directory}")


if __name__ == "__main__":
    main()
