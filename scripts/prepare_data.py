import argparse
from ml.datasets.abo import acquire
from scripts.dataset_cleaning import clean
from scripts.dataset_inspection import inspect
from scripts.dataset_split import split
from scripts.import_catalog import import_catalog
from scripts.dataset_visualization import visualize

def main():
    p=argparse.ArgumentParser(description="Acquire real ABO listings/images, validate them, and split by product identity.")
    p.add_argument("--max-products",type=int); p.add_argument("--max-images-per-product",type=int); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); report=acquire(a.max_products,a.max_images_per_product)
    clean(); stats=inspect()
    if stats["valid_images"]:
        split("data/processed/metadata/catalog.csv","data/splits",a.seed)
        import_catalog()
        visualize()
    print(f"Prepared {report['downloaded_valid_images']} valid real images across {report['identities_with_images']} identities.")
if __name__ == "__main__": main()
