# Benchmark Datasets

The benchmark scripts require three external datasets. Download instructions are provided below. Place all datasets under a `data/` directory at the project root (this directory is excluded from version control via `.gitignore`).

---

## ShapeNet (classification benchmark)

**Script:** `conformal_features/benchmarks/shapenet_classify.py`

**Download:**
- Register at https://shapenet.org and accept the terms of use.
- Download ShapeNetCore.v2: https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip (~25 GB)
- Alternatively, use the ShapeNet Part dataset (smaller) from https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

**Expected directory structure:**
```
data/
  ShapeNetCore.v2/
    02691156/   # airplane
      <model_id>/
        models/
          model_normalized.obj
    02958343/   # car
    ...
```

**Usage:**
```bash
conformal-shapenet --data data/ShapeNetCore.v2 --categories airplane,car,chair
```

---

## SHREC (retrieval benchmark)

**Script:** `conformal_features/benchmarks/shrec_retrieval.py`

**Download:**
- SHREC'11 Non-rigid 3D Shape Retrieval: https://www.shrec.net/
- Direct mirror (SHREC'11): http://tosca.cs.technion.ac.il/data/shrec11.zip

**Expected directory structure:**
```
data/
  shrec11/
    models/
      <class_id>/
        <model_id>.off
    classification.txt
```

**Usage:**
```bash
conformal-shrec --data data/shrec11
```

---

## FAUST (correspondence benchmark)

**Script:** `conformal_features/benchmarks/faust_correspondence.py`

**Download:**
- Register at https://faust-leaderboard.is.tuebingen.mpg.de/ to obtain access.
- Download the MPI-FAUST dataset: https://faust-leaderboard.is.tuebingen.mpg.de/downloads

**Expected directory structure:**
```
data/
  MPI-FAUST/
    training/
      registrations/
        tr_reg_000.ply
        tr_reg_001.ply
        ...
        tr_reg_099.ply
      challenge/
        ...
```

**Usage:**
```bash
conformal-faust --data data/MPI-FAUST
```

---

## Notes

- None of these datasets are redistributed with this repository. You must download them separately and agree to each dataset's individual terms of use.
- The `data/` directory is listed in `.gitignore` and will not be committed.
- Approximate disk requirements: ShapeNet ~25 GB, SHREC ~200 MB, FAUST ~300 MB.
