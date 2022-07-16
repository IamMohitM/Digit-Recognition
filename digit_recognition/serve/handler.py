from ts.torch_handler.base_handler import BaseHandler
import torch
from PIL import Image
from torchvision import transforms
import logging
from ast import literal_eval
import os
import base64


logger = logging.getLogger(__name__)


class CustomHandler(BaseHandler):
    def __init__(self):
        super(CustomHandler, self).__init__()
        self.image_processing = transforms.Compose([

            transforms.Resize((96, 96)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:

            
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(
                model_dir, model_file, model_pt_path)
            self.model.to(self.device)

            self.model.eval()
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")
            self.model = self._load_torchscript_model(model_pt_path)

        logger.debug('Model file %s loaded successfully', model_pt_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """
        return torch.jit.load(model_pt_path, map_location=self.device)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.

        Args:
            model_dir (str): Points to the location of the model artefacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.

        Raises:
            RuntimeError: It raises this error when the model.py file is missing.
            ValueError: Raises value error when there is more than one class in the label,
                        since the mapping supports only one label per class.

        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        from models import CNN
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        model = CNN()
        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.device)
            model.load_state_dict(state_dict)
        return model

    def preprocess(self, data):
        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data")

            shape = row.get('shape')
            shape = literal_eval(shape.decode("utf-8"))
            if isinstance(image, (bytearray, bytes)):
                image = torch.frombuffer(image, dtype=torch.float32)
                image = torch.reshape(image, shape)
                # image = Image.open(io.BytesIO(image)).convert("RGB")
                # image.save("test.png")
                logging.info(f"image shape before preprocess: {image.shape}")
                logging.info(f"image type before preprocess: {type(image)}")
                logging.info(f"image dtype before preprocess: {image.dtype}")

                image = self.image_processing(image)
                logging.info(f"image shape after preprocess: {image.shape}")
                logging.info(f"image type after preprocess: {type(image)}")
                logging.info(f"image dtype after preprocess: {image.dtype}")

            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):
        logging.info("Inside Post Process")
        logging.info(f"Outputs: {data}")
        max_scores, predictions = torch.max(data, axis=1)
        predictions = predictions
        logging.info(max_scores)
        logging.info(predictions.tolist())
        return predictions.tolist()
