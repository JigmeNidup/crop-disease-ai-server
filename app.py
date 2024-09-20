from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
# from flask_ngrok import run_with_ngrok

#fruit models
print("Loading Fruit Models ....")
Apple_model = tf.keras.models.load_model('./models/Fruits/Apple_model.h5')
Banana_model = tf.keras.models.load_model('./models/Fruits/Banana_model.h5')
Citrus_model = tf.keras.models.load_model('./models/Fruits/Citrus_model.h5')
Grapes_model = tf.keras.models.load_model('./models/Fruits/Grapes_model.h5')
Mango_model = tf.keras.models.load_model('./models/Fruits/Mango_model.h5')
print("Loaded Fruit Models!")

#fruit models
print("Loading Crop Models ....")
Corn_model = tf.keras.models.load_model('./models/Crops/Corn_model.h5')
Potato_model = tf.keras.models.load_model('./models/Crops/Potato_model.h5')
Rice_model = tf.keras.models.load_model('./models/Crops/Rice_model.h5')
Tomato_model = tf.keras.models.load_model('./models/Crops/Tomato_model.h5')
Wheat_model = tf.keras.models.load_model('./models/Crops/Wheat_model.h5')
print("Loaded Crop Models!")

app = Flask(__name__)
# run_with_ngrok(app)   
cors = CORS(app)

def preprocess_base64_image(base64_string):
    # Convert base64 string to bytes
    img_data = base64.b64decode(base64_string)
    
    # Read image from bytes
    img = Image.open(BytesIO(img_data))
    
    img = img.resize((32, 32))
    img = img.convert('RGB')
    
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def hello_world():
	return 'Hello World'


@app.route('/image',methods=['POST'])
def process_image():
    try:
        data = request.json
        crop = data["type"]
        name = data["name"]

        input_image = preprocess_base64_image(data["filebase64"][23:])
        if (crop == "Fruits"):
            if(name == "Apple"):
                class_names = ['Alternaria Leaf Spot','Apple rot','Apple Scab','Block rot','Brown Spot','Cedar apple rust','Frogeye Leaf Spot','Grey Spot','Healthy','Leaf Blotch','Mosaic','Powdery Mildew','Rust']
                predictions = Apple_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Banana"):
                class_names = ['Banana Black Sigatoka Disease', 'Banana Bract Mosaic Virus Disease', 'Banana Healthy Leaf', 'Banana Insect Pest Disease', 'Banana Moko Disease','Banana Panama Disease','Banana Yellow Sigatoka Disease']
                predictions = Banana_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Citrus"):
                class_names = ['Black spot','Canker','Greening','Healthy','Melanose']
                predictions = Citrus_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Grapes"):
                class_names = ['Black_Measles','Black_rot','Healthy','Isariopsis_Leaf_Spot']
                predictions = Grapes_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Mango"):
                class_names = ['Anthracnose','Bacterial Canker','Cutting Weevil','Die Back','Gall Midge','Healthy','Powdery Mildew','Sooty Mould']
                predictions = Mango_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
        elif(crop == "Crops"):
            if(name == "Corn"):
                class_names = ['Common_Rust','Gray_Leaf_Spot','Healthy','Northern_Leaf_Blight']
                predictions = Corn_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Potato"):
                class_names = ['Early_blight','Healthy','Late_blight']
                predictions = Potato_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Rice"):
                class_names = ['Brown_Spot','Healthy','Leaf_Blast','Neck_Blast']
                predictions = Rice_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Tomato"):
                class_names = ['Bacterial_spot','Early_blight','Healthy','Late_blight','Leaf_Mold','Septoria_leaf_spot','Spider_mites Two-spotted_spider_mite','Target_Spot','Tomato_mosaic_virus','Tomato_Yellow_Leaf_Curl_Virus']
                predictions = Tomato_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
            elif(name == "Wheat"):
                class_names = ['Brown_Rust','Healthy','Yellow_Rust']
                predictions = Wheat_model.predict(input_image)
                predicted_class = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class]
                print("Predicted class:", predicted_class_name)
                return {"status":"true", "name":predicted_class_name}
    except error:
        return {"status:":"false","error":error}

app.run()