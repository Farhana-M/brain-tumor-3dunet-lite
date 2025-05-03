import os
import zipfile
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gdown
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

 
# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net - (Lightweight Architecture on Normal CPUs)")
 
# Function to download the default model from Google Drive
def download_default_model():
    # Google Drive file ID for the default model
    file_id = "1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"  # replace with yours
    output_path = "default_model.keras"

    # Download only if it doesn’t already exist
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    return output_path
 
# Load the default model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_default_model():
    # Download the model from Google Drive (uses your download_default_model())
    model_path = download_default_model()
    model = load_model(model_path, compile=False)
    return model

# actually load it once, and Streamlit will cache it
default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path):
    """
    Load a 3D NIfTI file and min–max normalize all voxels to [0,1].
    """
    # Load as float32
    image = nib.load(file_path).get_fdata().astype(np.float32)  # shape (H,W,D)
    
    # Flatten to (N_voxels,1), fit and apply scaler, then reshape back
    flat = image.ravel().reshape(-1, 1)
    scaler = MinMaxScaler()
    flat_scaled = scaler.fit_transform(flat)
    image_norm = flat_scaled.reshape(image.shape)
    
    return image_norm 

# Function to combine 4 channels into a single 4-channel numpy array
def combine_channels(t1n, t1c, t2f, t2w):
    """
    Stack four 3D modalities into one array of shape (H, W, D, 4),
    then crop to the central region.
    """
    # Stack along last axis → (H,W,D,4)
    combined = np.stack([t1n, t1c, t2f, t2w], axis=-1)
    # Crop to the central 128×128×128 region
    combined = combined[56:184, 56:184, 13:141, :]
    
    return combined
 
# Function to run segmentation
def run_segmentation(model, input_image):
    """
    Given a model and a single 4-channel volume (H,W,D,4),
    run a prediction and return the argmax mask (H,W,D).
    """
    # Expand to batch dimension → (1,H,W,D,4)
    batch = np.expand_dims(input_image, axis=0)
    
    # Check shape
    if batch.ndim != 5 or batch.shape[-1] != 4:
        st.error(
            f"Unexpected shape for input_image: {batch.shape}. "
            "Expected (batch, height, width, depth, 4 channels)."
        )
        return None
    
    # 3) Run inference
    pred = model.predict(batch, verbose=0)            # shape (1,H,W,D,4)
    mask = np.argmax(pred[0], axis=-1).astype(np.uint8)  # shape (H,W,D)
    
    return mask
 
# Sidebar: model upload
st.sidebar.header("Upload Your Own Model")
uploaded_model = st.sidebar.file_uploader(
    "Upload a Keras model (.keras)",
    type=["keras"]
)

# Load the model (default or uploaded)
if uploaded_model is not None:
    # Save the uploaded model temporarily
    temp_path = "temp_model.keras"
    with open(temp_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    # Attempt to load the custom model
    try:
        model = load_model(temp_path, compile=False)
        st.sidebar.success("Custom model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading custom model: {e}")
        st.sidebar.info("Using the default model instead.")
        model = default_model
else:
    model = default_model
    st.sidebar.info("Using the default model.")

# Main app: NIfTI folder upload
st.header("Upload a Folder Containing NIfTI Files")
uploaded_folder = st.file_uploader(
    "Upload a folder (as a zip) containing T1n, T1c, T2f, T2w NIfTI files",
    type=["zip"]
)

if uploaded_folder is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save and extract the zip file
        zip_path = os.path.join(temp_dir, "uploaded_folder.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_folder.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Locate the required files
        t1n_path = t1c_path = t2f_path = t2w_path = mask_path = None
        for root, _, files in os.walk(temp_dir):
            for fname in files:
                if fname.endswith("t1n.nii.gz"):
                    t1n_path = os.path.join(root, fname)
                elif fname.endswith("t1c.nii.gz"):
                    t1c_path = os.path.join(root, fname)
                elif fname.endswith("t2f.nii.gz"):
                    t2f_path = os.path.join(root, fname)
                elif fname.endswith("t2w.nii.gz"):
                    t2w_path = os.path.join(root, fname)
                elif fname.endswith("seg.nii.gz"):
                    mask_path = os.path.join(root, fname)

        # Check presence of all channels
        if all([t1n_path, t1c_path, t2f_path, t2w_path]):
            # Preprocess and combine channels
            t1n = preprocess_nifti(t1n_path)
            t1c = preprocess_nifti(t1c_path)
            t2f = preprocess_nifti(t2f_path)
            t2w = preprocess_nifti(t2w_path)
            combined = combine_channels(t1n, t1c, t2f, t2w)

            st.write(f"Shape of combined_image: {combined.shape}")
            if combined.ndim != 4 or combined.shape[-1] != 4:
                st.error(
                    f"Unexpected shape: {combined.shape}. "
                    "Expected (height, width, depth, 4 channels)."
                )
            else:
                st.write("Running segmentation...")
                seg_result = run_segmentation(model, combined)
                st.write("Segmentation completed! Displaying results...")

                # Load and process ground-truth mask if available
                mask_argmax = None
                if mask_path:
                    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
                    mask[mask == 4] = 3  # consolidate label 4 into 3
                    mask_argmax = np.argmax(
                        to_categorical(mask, num_classes=4),
                        axis=3
                    )

                # Visualization
                slice_indices = [75, 90, 100]
                fig, axes = plt.subplots(3, 4, figsize=(18, 12))
                for i, sl in enumerate(slice_indices):
                    img_slice = np.rot90(combined[:, :, sl, 0])
                    pred_slice = np.rot90(seg_result[:, :, sl])

                    axes[i, 0].imshow(img_slice, cmap="gray")
                    axes[i, 0].set_title(f"Image - Slice {sl}")

                    if mask_argmax is not None:
                        gt_slice = np.rot90(mask_argmax[:, :, sl])
                        axes[i, 1].imshow(gt_slice)
                        axes[i, 1].set_title(f"GT - Slice {sl}")
                    else:
                        axes[i, 1].axis("off")

                    axes[i, 2].imshow(pred_slice)
                    axes[i, 2].set_title(f"Prediction - Slice {sl}")

                    axes[i, 3].imshow(img_slice, cmap="gray")
                    axes[i, 3].imshow(pred_slice, alpha=0.5)
                    axes[i, 3].set_title(f"Overlay - Slice {sl}")

                plt.tight_layout()
                st.pyplot(fig)

                # Save & offer download of the result
                out_file = "segmentation_result.nii.gz"
                nib.save(
                    nib.Nifti1Image(seg_result.astype(np.float32), np.eye(4)),
                    out_file
                )
                with open(out_file, "rb") as f:
                    st.download_button(
                        label="Download Segmentation Result",
                        data=f,
                        file_name=out_file,
                        mime="application/octet-stream"
                    )
                os.remove(out_file)
        else:
            st.error(
                "The uploaded folder must contain T1n, T1c, T2f and T2w NIfTI files."
            )
