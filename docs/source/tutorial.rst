Tutorial


.. admonition:: Tutorial Preliminaries: Data Preparation and Helpers
  :class: toggle

  To follow the example here, install those packages::

    uv pip install volara

  .. jupyter-execute::

    from funlib.persistence import open_ds, prepare_ds
    from skimage import data

    cell_data = data.cells3d()

    array = prepare_ds("cells3d.zarr", offset=(0,0,0), voxel_size=(290,260,260))
    # f = zarr.open('sample_data.zarr', 'w')
    # f['raw'] = raw_data
    # f['raw'].attrs['resolution'] = (1, 1)
    # f['ground_truth'] = gt_data
    # f['ground_truth'].attrs['resolution'] = (1, 1)

ahhh