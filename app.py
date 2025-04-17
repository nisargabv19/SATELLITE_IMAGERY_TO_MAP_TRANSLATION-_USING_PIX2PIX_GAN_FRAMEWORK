#
# from flask import Flask, request, render_template, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import base64
#
# app = Flask(__name__)
#
# # Load the saved model
# model = tf.keras.models.load_model('pix2pix_generator_old.h5')
#
#
# def preprocess_image(image):
#     # Convert RGBA to RGB if needed
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')
#
#     # Resize image to 256x256
#     image = image.resize((256, 256))
#
#     # Convert to numpy array and normalize
#     image = np.array(image)
#
#     # Ensure we have 3 channels
#     if len(image.shape) == 2:  # If grayscale, convert to RGB
#         image = np.stack((image,) * 3, axis=-1)
#     elif image.shape[-1] == 4:  # If RGBA, convert to RGB
#         image = image[:, :, :3]
#
#     # Normalize to [-1, 1]
#     image = (image / 127.5) - 1
#
#     # Add batch dimension
#     image = np.expand_dims(image, axis=0)
#     return image
#
#
# def postprocess_image(generated_image):
#     # If tensor, convert to numpy
#     if isinstance(generated_image, tf.Tensor):
#         generated_image = generated_image.numpy()
#
#     # Convert from [-1, 1] to [0, 255]
#     generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
#
#     # Ensure we're working with the first image if it's a batch
#     if len(generated_image.shape) == 4:
#         generated_image = generated_image[0]
#
#     # Convert to PIL Image
#     generated_image = Image.fromarray(generated_image)
#     return generated_image
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         # Get the image from the POST request
#         file = request.files['image']
#         if not file:
#             return jsonify({'success': False, 'error': 'No file uploaded'})
#
#         # Open and verify the image
#         try:
#             image = Image.open(file.stream)
#         except Exception as e:
#             return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'})
#
#         # Preprocess the image
#         try:
#             processed_image = preprocess_image(image)
#         except Exception as e:
#             return jsonify({'success': False, 'error': f'Error preprocessing image: {str(e)}'})
#
#         # Generate the output using the model
#         try:
#             generated_image = model.predict(processed_image)
#         except Exception as e:
#             return jsonify({'success': False, 'error': f'Error during model prediction: {str(e)}'})
#
#         # Postprocess the generated image
#         try:
#             output_image = postprocess_image(generated_image)
#         except Exception as e:
#             return jsonify({'success': False, 'error': f'Error postprocessing image: {str(e)}'})
#
#         # Convert to base64 for sending back to frontend
#         try:
#             buffered = io.BytesIO()
#             output_image.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode()
#         except Exception as e:
#             return jsonify({'success': False, 'error': f'Error converting image to base64: {str(e)}'})
#
#         return jsonify({'success': True, 'image': img_str})
#
#     except Exception as e:
#         return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

# from flask import Flask, request, render_template, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import base64
# import logging
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
#
# # Load the saved model
# try:
#     model = tf.keras.models.load_model('pix2pix_generator_old.h5')
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading model: {str(e)}")
#     raise
#
#
# def preprocess_image(image):
#     """
#     Preprocess the input image for the model.
#     """
#     try:
#         # Log original image details
#         logger.info(f"Original image size: {image.size}, mode: {image.mode}")
#
#         # Convert RGBA to RGB if needed
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')
#             logger.info("Converted RGBA to RGB")
#
#         # Convert grayscale to RGB if needed
#         if image.mode == 'L':
#             image = image.convert('RGB')
#             logger.info("Converted grayscale to RGB")
#
#         # Resize image to 256x256
#         image = image.resize((256, 256), Image.Resampling.LANCZOS)
#         logger.info("Resized image to 256x256")
#
#         # Convert to numpy array
#         image_array = np.array(image)
#         logger.info(f"Converted to numpy array, shape: {image_array.shape}")
#
#         # Ensure we have 3 channels
#         if len(image_array.shape) == 2:
#             image_array = np.stack((image_array,) * 3, axis=-1)
#             logger.info("Expanded grayscale to RGB")
#         elif image_array.shape[-1] == 4:
#             image_array = image_array[:, :, :3]
#             logger.info("Removed alpha channel")
#
#         # Log value ranges before normalization
#         logger.info(f"Value range before normalization: min={image_array.min()}, max={image_array.max()}")
#
#         # Normalize to [-1, 1]
#         image_array = (image_array / 127.5) - 1
#         logger.info(f"Normalized value range: min={image_array.min()}, max={image_array.max()}")
#
#         # Add batch dimension
#         image_array = np.expand_dims(image_array, axis=0)
#         logger.info(f"Final preprocessed shape: {image_array.shape}")
#
#         return image_array
#
#     except Exception as e:
#         logger.error(f"Error in preprocess_image: {str(e)}")
#         raise
#
#
# def postprocess_image(generated_image):
#     """
#     Postprocess the generated image for display.
#     """
#     try:
#         # Log input details
#         logger.info(f"Generated image type: {type(generated_image)}, shape: {generated_image.shape}")
#
#         # If tensor, convert to numpy
#         if isinstance(generated_image, tf.Tensor):
#             generated_image = generated_image.numpy()
#             logger.info("Converted tensor to numpy array")
#
#         # Log value ranges before denormalization
#         logger.info(f"Value range before denormalization: min={generated_image.min()}, max={generated_image.max()}")
#
#         # Convert from [-1, 1] to [0, 255]
#         generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
#         logger.info(f"Denormalized value range: min={generated_image.min()}, max={generated_image.max()}")
#
#         # Ensure we're working with the first image if it's a batch
#         if len(generated_image.shape) == 4:
#             generated_image = generated_image[0]
#             logger.info("Removed batch dimension")
#
#         # Convert to PIL Image
#         output_image = Image.fromarray(generated_image)
#         logger.info(f"Final image size: {output_image.size}, mode: {output_image.mode}")
#
#         return output_image
#
#     except Exception as e:
#         logger.error(f"Error in postprocess_image: {str(e)}")
#         raise
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         # Get the image from the POST request
#         if 'image' not in request.files:
#             logger.error("No image file in request")
#             return jsonify({'success': False, 'error': 'No file uploaded'})
#
#         file = request.files['image']
#         if not file:
#             logger.error("Empty file object")
#             return jsonify({'success': False, 'error': 'Empty file'})
#
#         # Open and verify the image
#         try:
#             image = Image.open(file.stream)
#             logger.info(f"Received image: size={image.size}, mode={image.mode}")
#         except Exception as e:
#             logger.error(f"Error opening image: {str(e)}")
#             return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'})
#
#         # Preprocess the image
#         try:
#             processed_image = preprocess_image(image)
#         except Exception as e:
#             logger.error(f"Preprocessing error: {str(e)}")
#             return jsonify({'success': False, 'error': f'Error preprocessing image: {str(e)}'})
#
#         # Generate the output using the model
#         try:
#             logger.info("Starting model prediction")
#             generated_image = model(processed_image, training=False)
#             logger.info("Model prediction completed")
#         except Exception as e:
#             logger.error(f"Model prediction error: {str(e)}")
#             return jsonify({'success': False, 'error': f'Error during model prediction: {str(e)}'})
#
#         # Postprocess the generated image
#         try:
#             output_image = postprocess_image(generated_image)
#         except Exception as e:
#             logger.error(f"Postprocessing error: {str(e)}")
#             return jsonify({'success': False, 'error': f'Error postprocessing image: {str(e)}'})
#
#         # Convert to base64 for sending back to frontend
#         try:
#             buffered = io.BytesIO()
#             output_image.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode()
#             logger.info("Successfully converted to base64")
#         except Exception as e:
#             logger.error(f"Base64 conversion error: {str(e)}")
#             return jsonify({'success': False, 'error': f'Error converting image to base64: {str(e)}'})
#
#         return jsonify({'success': True, 'image': img_str})
#
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import logging
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the saved model
try:
    model = load_model('pix2pix_generator.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


def preprocess_image(image):
    """
    Preprocess the input image for the model.
    - Converts to RGB if necessary.
    - Resizes to 256x256.
    - Normalizes pixel values to [-1, 1].
    """
    try:
        logger.info(f"Original image size: {image.size}, mode: {image.mode}")

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode.")

        # Resize to 256x256
        image = image.resize((256, 256), Image.Resampling.BICUBIC)
        logger.info(f"Resized image to 256x256.")

        # Convert to numpy array and normalize
        image_array = (np.array(image).astype(np.float32) / 127.5) - 1  # Normalize to [-1, 1]
        logger.info(f"Normalized image, shape: {image_array.shape}, range: ({image_array.min()}, {image_array.max()})")

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise


def postprocess_image(generated_image):
    """
    Postprocess the model's generated image for display.
    - Denormalizes pixel values from [-1, 1] to [0, 255].
    - Converts numpy array to a PIL Image.
    """
    try:
        # Convert EagerTensor to NumPy array
        generated_image = generated_image.numpy()  # Convert to NumPy array

        # Denormalize the image
        generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
        logger.info(f"Denormalized image to uint8, range: ({generated_image.min()}, {generated_image.max()})")

        # Remove batch dimension if present
        if generated_image.ndim == 4:
            generated_image = generated_image[0]
            logger.info("Removed batch dimension.")

        # Convert to PIL image
        output_image = Image.fromarray(generated_image)
        logger.info(f"Postprocessed image, size: {output_image.size}, mode: {output_image.mode}")
        return output_image

    except Exception as e:
        logger.error(f"Error in postprocess_image: {str(e)}")
        raise


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate a satellite view from the input map using the model.
    - Accepts an uploaded image via POST request.
    - Returns the generated image as base64-encoded JSON.
    """
    try:
        # Check if the image file is in the request
        if 'image' not in request.files:
            logger.error("No image file in request.")
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['image']
        if not file or not file.filename:
            logger.error("Empty or missing file.")
            return jsonify({'success': False, 'error': 'Empty file'})

        # Validate the file format
        if not (file.filename.lower().endswith('.png') or file.filename.lower().endswith('.jpg') or file.filename.lower().endswith('.jpeg')):
            logger.error("Invalid file format.")
            return jsonify({'success': False, 'error': 'Invalid file format. Only PNG and JPG are supported.'})

        # Open the image
        try:
            image = Image.open(file.stream)
            logger.info(f"Received image: size={image.size}, mode={image.mode}")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'})

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Generate the satellite view
        logger.info("Starting model prediction.")
        generated_image = model(processed_image, training=True)  # Set training=True to match Colab behavior
        logger.info("Model prediction completed.")

        # Postprocess the generated image
        output_image = postprocess_image(generated_image)

        # Convert to base64 for returning to the frontend
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.info("Successfully converted image to base64.")

        return jsonify({'success': True, 'image': img_str})

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
