import argparse, json
from ml.datasets.abo import acquire

def main():
    parser = argparse.ArgumentParser(description="Acquire a real handbag subset from the official ABO dataset.")
    parser.add_argument("--max-products", type=int, help="Optional reproducible cap for a smaller local run")
    parser.add_argument("--max-images-per-product", type=int, help="Optional cap; defaults to all listed images")
    args = parser.parse_args()
    print(json.dumps(acquire(args.max_products, args.max_images_per_product), indent=2))

if __name__ == "__main__": main()
