# Refined ABO handbag dataset audit

Scope: ABO handbags, purses, fashion totes, shoulder bags, clutches and crossbody bags; source product type/title evidence required

Input rows reviewed: 855  
Valid retained catalog images: 248  
Retained ABO product identities: 66  
Image records are source catalog images/views; the report does not claim separate real-world photo sessions.

## Retained categories

| Source product type | Images | Product identities |
|---|---:|---:|
| HANDBAG | 211 | 56 |
| BAG | 37 | 10 |

## Excluded rows

| Source type / validation | Excluded image rows |
|---|---:|
| BACKPACK | 184 |
| SUITCASE | 263 |
| LUGGAGE | 52 |
| BAG | 16 |
| CAMERA_BAGS_AND_CASES | 18 |
| WALLET | 55 |
| TOTE_BAG | 19 |

## Duplicate hashes and identity counts

Duplicate-hash groups: 0. No exact duplicate image hashes were found in the retained set.
Identities with fewer than two images: 2.

| ABO product identity | Catalog images |
|---|---:|
| `ABO:amazon.ae:B06XPGBJYR` | 2 |
| `ABO:amazon.ae:B06XPGTKD8` | 4 |
| `ABO:amazon.ae:B06XPLJ4LT` | 4 |
| `ABO:amazon.ae:B071LSJ8Z2` | 4 |
| `ABO:amazon.ae:B074CSVG97` | 4 |
| `ABO:amazon.ae:B078T86FRV` | 4 |
| `ABO:amazon.ae:B078T95WPP` | 4 |
| `ABO:amazon.ca:B00W4ZH2S2` | 4 |
| `ABO:amazon.ca:B014492Y9W` | 4 |
| `ABO:amazon.ca:B0144KKUBA` | 4 |
| `ABO:amazon.ca:B06VT9HX59` | 4 |
| `ABO:amazon.ca:B06VTHSQ8Z` | 4 |
| `ABO:amazon.ca:B06VTJJ6M8` | 4 |
| `ABO:amazon.ca:B06W2HCGQ2` | 4 |
| `ABO:amazon.ca:B06W53FBKV` | 4 |
| `ABO:amazon.ca:B06W9GMKN1` | 4 |
| `ABO:amazon.ca:B06X17B2N4` | 4 |
| `ABO:amazon.ca:B06XPF4QKN` | 4 |
| `ABO:amazon.ca:B06XPK3M4R` | 2 |
| `ABO:amazon.ca:B06XPKX2TR` | 4 |
| `ABO:amazon.ca:B06XRKCR12` | 4 |
| `ABO:amazon.ca:B06Y2XX9D1` | 3 |
| `ABO:amazon.ca:B071J6X58M` | 4 |
| `ABO:amazon.ca:B074CSNFT8` | 4 |
| `ABO:amazon.ca:B074CSVG98` | 4 |
| `ABO:amazon.ca:B074CT3YPN` | 4 |
| `ABO:amazon.ca:B074CT8NTF` | 1 |
| `ABO:amazon.ca:B074CTDQPS` | 1 |
| `ABO:amazon.ca:B074DY3VVS` | 4 |
| `ABO:amazon.ca:B074F5Z131` | 4 |
| `ABO:amazon.ca:B074ZV53RM` | 4 |
| `ABO:amazon.ca:B07574KZYH` | 4 |
| `ABO:amazon.ca:B075751SP2` | 4 |
| `ABO:amazon.ca:B075753XF1` | 4 |
| `ABO:amazon.ca:B07669CSMG` | 4 |
| `ABO:amazon.ca:B0766CCYC9` | 4 |
| `ABO:amazon.ca:B0767GSBRG` | 4 |
| `ABO:amazon.ca:B0767VBGCX` | 4 |
| `ABO:amazon.ca:B0767VBGDK` | 4 |
| `ABO:amazon.ca:B0767Y1MQD` | 4 |
| `ABO:amazon.ca:B0767ZD7CS` | 4 |
| `ABO:amazon.ca:B078LY1FWN` | 4 |
| `ABO:amazon.ca:B078M6S331` | 4 |
| `ABO:amazon.ca:B078TB372H` | 4 |
| `ABO:amazon.ca:B0792CYTGJ` | 4 |
| `ABO:amazon.ca:B0792DDJPH` | 4 |
| `ABO:amazon.ca:B0792FNKWS` | 4 |
| `ABO:amazon.ca:B0792LQ3KS` | 4 |
| `ABO:amazon.ca:B07QQ3GPNT` | 4 |
| `ABO:amazon.ca:B07QSCSH5T` | 4 |
| `ABO:amazon.ca:B0843C7WYG` | 4 |
| `ABO:amazon.ca:B084THL8Y2` | 4 |
| `ABO:amazon.ca:B089J6J66V` | 4 |
| `ABO:amazon.ca:B089K3RZXQ` | 3 |
| `ABO:amazon.ca:B089K4TYX1` | 3 |
| `ABO:amazon.ca:B089KF2ZW8` | 4 |
| `ABO:amazon.ca:B089KJMXYY` | 4 |
| `ABO:amazon.ca:B08F9ZWWXP` | 3 |
| `ABO:amazon.ca:B08FB1QPRF` | 3 |
| `ABO:amazon.ca:B08FB4X8KG` | 3 |
| `ABO:amazon.co.jp:B00JM5VN9G` | 4 |
| `ABO:amazon.co.jp:B084Y5C4LC` | 4 |
| `ABO:amazon.co.uk:B01KP0P1X0` | 4 |
| `ABO:amazon.co.uk:B0749WPZCT` | 4 |
| `ABO:amazon.co.uk:B07D7Y25MT` | 4 |
| `ABO:amazon.co.uk:B07D7YL6RF` | 4 |

## Identity-disjoint split

| Partition | Product identities |
|---|---:|
| train | 52 |
| validation | 7 |
| test | 7 |

Verified intersections: train∩validation = 0; train∩test = 0; validation∩test = 0.
Identities with fewer than two images: `ABO:amazon.ca:B074CT8NTF`, `ABO:amazon.ca:B074CTDQPS`.

Seed: 42. Detailed machine-readable report: `data/reports/handbag_dataset_audit.json` (local artifact).
