## XSA Access

Some functions to retrieve data from [XMM-Newton Science Archive](https://nxsa.esac.esa.int/). It relies mostly on [astropy](https://docs.astropy.org/en/stable/io/unified.html#getting-started-with-table-i-o) and, especially, [astroquery.esa.xmm_newton](https://astroquery.readthedocs.io/en/latest/esa/xmm_newton/xmm_newton.html).

::: trapallada.xsa.access

## Some already identified light curves

| Obs.ID      | Target            | Source type          | Characteristic          |
|-------------|-------------------|----------------------|-------------------------|
| 0551120401  | Proxima Centauri | UV Ceti              | Flare                   |
| 0605950101  | EI Cha            | Transitional disk   | Increase                |
|             | EL Cha            |                      | Flare                   |
|             | RS Cha            | Eclipsing A binary  | Flare                   |
|             | EN Cha            | Transitional disk   | Decay phase             |
|             | EP Cha            | Class II             | Double flare            |
| 0891801301  | AD Leo            | UV Ceti              | Two flares              |
| 0205770101  | Z Chamaleontis    | Eclipsing Nova       | Eclipse                 |
| 0652480201  | XY Ari            | Cataclysmic binary  | Eclipse                 |
| 0160760201  | EXO 0748-676      | LMXB                 | Bursts with orbital modulation |

# File names

[XMM-Newton Data Files Handbook](http://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/dfhb/)

In `P0605950101PNX000SRCTSR8001.FTZ` the filename encodes the following information:

- **Observation ID**: `0605950101`
  - `P` is a prefix indicating it is a public observation.
  - `0605950101` is the unique identification number for the observation.
  
- **Instrument**: `PN`
  - `PN` stands for the EPIC-pn camera.
  - Other possible instruments include `MOS1`, `MOS2`, `RGS1`, `RGS2`, and `OM`.
  
- **Exposure/Sequence Number**: `X000`
  - `X` could be a placeholder or indicate a specific processing type.
  - `000` represents the exposure number or sequence.

- **Type of File**: `SRC`
  - `SRC` stands for source list or source-specific data.

- **Data Product/Processing Stage**: `TSR8`
  - `TSR` could be a code related to the specific type of analysis or data product.
  - `8` is often a version number or a further specification.

- **File Identifier**: `001`
  - `001` indicates it is the first in a series.

- **File Format**: `FTZ`
  - `FTZ` indicates that the file is in FITS (Flexible Image Transport System) format and is compressed.
  - If the file were not compressed, it would have the extension `FIT`.
