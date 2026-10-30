import argparse
from ml.datasets.abo import acquire
from scripts.dataset_cleaning import clean
from scripts.dataset_inspection import inspect
from scripts.import_catalog import import_catalog
from scripts.dataset_visualization import visualize
from scripts.refine_handbag_dataset import refine

def main():
    p=argparse.ArgumentParser(description="Acquire real ABO listings/images, validate them, and split by product identity.")
    p.add_argument("--max-products",type=int); p.add_argument("--max-images-per-product",type=int); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); report=acquire(a.max_products,a.max_images_per_product)
    clean(); refine(seed=a.seed,snapshot=True); stats=inspect()
    if stats["valid_images"]:
        import_catalog()
        visualize()
    print(f"Prepared {stats['valid_images']} valid handbag-scope catalog images across {stats['bag_product_identities']} product identities.")
if __name__ == "__main__": main()
