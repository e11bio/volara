# %% [markdown]
# # Cremi example
# This example shows how to use volara to predict LSDs and affinities on the cremi dataset, and then run mutex watershed on the predicted affinities.

# %%
from pathlib import Path

import wget
from funlib.geometry import Coordinate

# Download some cremi data
# immediately convert it to zarr for convenience
if not Path("sample_A+_20160601.zarr").exists():
    wget.download(
        "https://cremi.org/static/data/sample_A+_20160601.hdf", "sample_A+_20160601.hdf"
    )
if not Path("sample_A+_20160601.zarr/raw").exists():
    import h5py
    import zarr

    raw_ds = zarr.open("sample_A+_20160601.zarr", "w").create_dataset(
        "raw", data=h5py.File("sample_A+_20160601.hdf", "r")["volumes/raw"][:]
    )
    raw_ds.attrs["voxel_size"] = (40, 4, 4)
    raw_ds.attrs["axis_names"] = ["z", "y", "x"]
    raw_ds.attrs["unit"] = ["nm", "nm", "nm"]

# %% [markdown]
# Now we can predict the LSDs and affinities for this dataset. We have provided a very simple
# pretrained model for this dataset. We went for speed and efficiency over accuracy for this
# model so that it can run in a github action. You can train a significantly better model
# with access to a GPU and more Memmory.

# %%
# Here are some important details about the model:

# The number of output channels of our model. 10 lsds, 7 affinities
out_channels = [10, 7]

# The input shape of our model (not including channels)
min_input_shape = Coordinate(36, 252, 252)

# The output shape of our model (not including channels)
min_output_shape = Coordinate(32, 160, 160)

# The minimum increment for adjusting the input shape
min_step_shape = Coordinate(1, 1, 1)

# The range of predicted values. We have a sigmoid activation on our model
out_range = (0, 1)

# How much to grow the input shape for prediction. This is usually adjusted to maximize GPU memory,
# but depends on how you saved your model. The model we provided does not support different
# input shapes.
pred_size_growth = Coordinate(0, 0, 0)


# %%
from volara.blockwise import Predict
from volara.datasets import Affs, Raw
from volara.models import TorchModel

# %% [markdown]

# First we define the datasets that we are using along with some basic information about them

# %%
# our raw data is stored in uint8, but our model expects floats in range (0, 1) so we scale it
raw_dataset = Raw(store="sample_A+_20160601.zarr/raw", scale_shift=(1 / 255, 0))
# The affinities neighborhood depends on the model that was trained. Here we learned long range xy affinities
affs_dataset = Affs(
    store="sample_A+_20160601.zarr/affs",
    neighborhood=[
        Coordinate(1, 0, 0),
        Coordinate(0, 1, 0),
        Coordinate(0, 0, 1),
        Coordinate(0, 6, 0),
        Coordinate(0, 0, 6),
        Coordinate(0, 18, 0),
        Coordinate(0, 0, 18),
    ],
)
# We are just storing the lsds in a simple zarr dataset using the same format as the raw data
lsds_dataset = Raw(store="sample_A+_20160601.zarr/lsds")

# %% [markdown]
# Now we can define our model with the parameters we defined above. We will use the
# `TorchModel` class to load the model from a checkpoint and pass it to the `Predict` class.

# %%
torch_model = TorchModel(
    save_path="checkpoint_data/model.pt",
    checkpoint_file="checkpoint_data/model_checkpoint_15000",
    in_channels=1,
    out_channels=out_channels,
    min_input_shape=min_input_shape,
    min_output_shape=min_output_shape,
    min_step_shape=min_step_shape,
    out_range=out_range,
    pred_size_growth=pred_size_growth,
)
predict_cremi = Predict(
    checkpoint=torch_model,
    in_data=raw_dataset,
    out_data=[lsds_dataset, affs_dataset],
)

predict_cremi.run_blockwise(multiprocessing=False)

# %% [markdown]
# Let's visualize the results

# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(predict_cremi.in_data.array("r")[100])
ax[0].set_title("Raw")
ax[1].imshow(predict_cremi.out_data[0].array("r")[:3, 100].transpose(1, 2, 0))
ax[1].set_title("LSDs")
ax[2].imshow(predict_cremi.out_data[1].array("r")[3:6, 100].transpose(1, 2, 0))
ax[2].set_title("Affinities")
plt.show()

# %% [markdown]
# Now we can convert the results to a segmentation. We will run mutex watershed on the affinities in a multi step process.
# 1) Local fragment extraction - This step runs blockwise and generates fragments from the affinities. For each fragment we save a node in a graph with attributes such as its spatial position and size.
# 2) Edge extraction - This step runs blockwise and computes mean affinities between fragments, adding edges to the fragment graph.
# 3) Graph Mutex Watershed - This step runs on the fragment graph, and creates a lookup table from fragment id -> segment id.
# 4) Relabel fragments - This step runs blockwise and creates the final segmentation.

# %%
from volara.blockwise import AffAgglom, ExtractFrags, GraphMWS, Relabel
from volara.datasets import Labels
from volara.dbs import SQLite
from volara.lut import LUT

# %% [markdown]
# First lets define the graph and arrays we are going to use.

# because our graph is in an sql database, we need to define a schema with column names and types
# for node and edge attributes.
# For nodes: The defaults such as "id", "position", and "size" are already defined
# so we only need to define the additional attributes, in this case we have no additional node attributes.
# For edges: The defaults such as "id", "u", "v" are already defined, so we are only adding the additional
# attributes "xy_aff", "z_aff", and "lr_aff" for saving the mean affinities between fragments.

# %%
fragments_graph = SQLite(
    path="sample_A+_20160601.zarr/fragments.db",
    edge_attrs={"xy_aff": "float", "z_aff": "float", "lr_aff": "float"},
)
fragments_dataset = Labels(store="sample_A+_20160601.zarr/fragments")
segments_dataset = Labels(store="sample_A+_20160601.zarr/segments")

# %% [markdown]
# Now we define the tasks with the parameters we want to use.

# %%

# Generate fragments in blocks
extract_frags = ExtractFrags(
    db=fragments_graph,
    affs_data=affs_dataset,
    frags_data=fragments_dataset,
    block_size=min_output_shape,
    context=Coordinate(3, 6, 6) * 2,  # A bit larger than the longest affinity
    bias=[-0.6] + [-0.4] * 2 + [-0.6] * 2 + [-0.8] * 2,
    strides=(
        [Coordinate(1, 1, 1)] * 3
        + [Coordinate(1, 3, 3)] * 2  # We use larger strides for larger affinities
        + [Coordinate(1, 6, 6)] * 2  # This is to avoid excessive splitting
    ),
    randomized_strides=True,  # converts strides to probabilities of sampling affinities (1/prod(stride))
    remove_debris=64,  # remove excessively small fragments
    num_workers=4,
)

# Generate agglomerated edge scores between fragments via mean affinity accross all edges connecting two fragments
aff_agglom = AffAgglom(
    db=fragments_graph,
    affs_data=affs_dataset,
    frags_data=fragments_dataset,
    block_size=min_output_shape,
    context=Coordinate(3, 6, 6) * 1,
    scores={
        "z_aff": affs_dataset.neighborhood[0:1],
        "xy_aff": affs_dataset.neighborhood[1:3],
        "lr_aff": affs_dataset.neighborhood[3:],
    },
    num_workers=4,
)

# Run mutex watershed again, this time on the fragment graph with agglomerated edges
# instead of the voxel graph of affinities
lut = LUT(path="sample_A+_20160601.zarr/lut.npz")
total_roi = raw_dataset.array("r").roi
graph_mws = GraphMWS(
    db=fragments_graph,
    lut=lut,
    weights={"xy_aff": (1, -0.4), "z_aff": (1, -0.6), "lr_aff": (1, -0.6)},
    roi=(total_roi.offset, total_roi.shape),
)

# Relabel the fragments into segments
relabel = Relabel(
    lut=lut,
    frags_data=fragments_dataset,
    seg_data=segments_dataset,
    block_size=min_output_shape,
    num_workers=4,
)

pipeline = extract_frags + aff_agglom + graph_mws + relabel
pipeline.run_blockwise(multiprocessing=True)

# %% [markdown]
# Let's visualize
#
# If you are following through on your own, I highly recommend installing `funlib.show.neuroglancer`, and
# running the command line tool via `neuroglancer -d sample_A+_20160601.zarr/*` to visualize the results in
# neuroglancer.
#
# For the purposes of visualizing here, we will make a simple gif


# %%
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

fragments = fragments_dataset.array("r")[:, ::2, ::2]
segments = segments_dataset.array("r")[:, ::2, ::2]
raw = raw_dataset.array("r")[:, ::2, ::2]

# Get unique labels
unique_labels = set(np.unique(fragments)) | set(np.unique(segments))
num_labels = len(unique_labels)


def random_color(label):
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(label)))
    return np.array((rs.random(), rs.random(), rs.random()))


# Generate random colors for each label
random_fragment_colors = [random_color(label) for label in range(num_labels)]

# Create a colormap
cmap_labels = ListedColormap(random_fragment_colors)

# Map labels to indices for the colormap
label_to_index = {label: i for i, label in enumerate(unique_labels)}
indexed_fragments = np.vectorize(label_to_index.get)(fragments)
indexed_segments = np.vectorize(label_to_index.get)(segments)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

ims = []
for i, (raw_slice, fragments_slice, segments_slice) in enumerate(
    zip(raw, indexed_fragments, indexed_segments)
):
    # Show the raw data
    if i == 0:
        im_raw = axes[0].imshow(raw_slice)
        axes[0].set_title("Raw")
        im_fragments = axes[1].imshow(
            fragments_slice,
            cmap=cmap_labels,
            vmin=0,
            vmax=num_labels,
            interpolation="none",
        )
        axes[1].set_title("Fragments")
        im_segments = axes[2].imshow(
            segments_slice,
            cmap=cmap_labels,
            vmin=0,
            vmax=num_labels,
            interpolation="none",
        )
        axes[2].set_title("Segments")
    else:
        im_raw = axes[0].imshow(raw_slice, animated=True)
        im_fragments = axes[1].imshow(
            fragments_slice,
            cmap=cmap_labels,
            vmin=0,
            vmax=num_labels,
            interpolation="none",
            animated=True,
        )
        im_segments = axes[2].imshow(
            segments_slice,
            cmap=cmap_labels,
            vmin=0,
            vmax=num_labels,
            interpolation="none",
            animated=True,
        )
    ims.append([im_raw, im_fragments, im_segments])

ims = ims + ims[::-1]
ani = animation.ArtistAnimation(fig, ims, blit=True)
ani.save("_static/cremi/segmentation.gif", writer="pillow", fps=10)

# %% [markdown]
# The final segmentation is shown below. Obviously this is not a great segmentation, but it is
# reasonably good for a model small enough to process a CREMI dataset in 20 minutes on a github
# action.
# ![segmentation](_static/cremi/segmentation.gif)
