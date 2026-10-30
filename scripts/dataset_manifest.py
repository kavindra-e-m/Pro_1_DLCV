"""Create the project manifest by acquiring the filtered, license-attributed ABO subset."""
from ml.datasets.abo import acquire
if __name__ == "__main__": print(acquire())
