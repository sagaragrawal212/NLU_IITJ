data_link_dict = {
                   "All_Beauty" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Beauty.json.gz",
                  "AMAZON_FASHION" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz",
                   "Appliances" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Appliances.json.gz",
                   "Arts_Crafts_and_Sewing":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Arts_Crafts_and_Sewing.json.gz",
                   "Automotive":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Automotive.json.gz",
                   "Books":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz",
                   "CDs_and_Vinyl" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/CDs_and_Vinyl.json.gz",
                   "Cell_Phones_and_Accessories" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Cell_Phones_and_Accessories.json.gz",
                   "Clothing_Shoes_and_Jewelry" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz",
                   "Digital_Music":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Digital_Music.json.gz",
                   "Electronics":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Electronics.json.gz",
                   "Gift_Cards":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Gift_Cards.json.gz",
                   "Grocery_and_Gourmet_Food" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Grocery_and_Gourmet_Food.json.gz",
                   "Home_and_Kitchen" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Home_and_Kitchen.json.gz",
                  "Industrial_and_Scientific" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Industrial_and_Scientific.json.gz",
                  "Kindle_Store":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Kindle_Store.json.gz",
                  "Luxury_Beauty":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Luxury_Beauty.json.gz",
                  "Magazine_Subscriptions":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Magazine_Subscriptions.json.gz",
                  "Movies_and_TV":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz",
                  "Musical_Instruments" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Musical_Instruments.json.gz",
                  "Office_Products" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Office_Products.json.gz",
                  "Patio_Lawn_and_Garden" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Patio_Lawn_and_Garden.json.gz",
                  "Pet_Supplies" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Pet_Supplies.json.gz",
                  "Prime_Pantry" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Prime_Pantry.json.gz",
                   "Software" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Software.json.gz",
                  "Sports_and_Outdoors":"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Sports_and_Outdoors.json.gz",
                  "Tools_and_Home_Improvement" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Tools_and_Home_Improvement.json.gz",
                  "Toys_and_Games" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Toys_and_Games.json.gz",
                  "Video_Games" : "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Video_Games.json.gz"


}


from google.colab import files
import gzip
import json
import pandas as pd
from tqdm import tqdm
import gc
import random
import os

def gunzip_file(file_path, output_path):
    with gzip.open(file_path, 'rb') as gz_file:
        with open(output_path, 'wb') as output_file:
            output_file.write(gz_file.read())

def load_json_file(file_path):

  all_reviews_df = []
  # Open the file and load the JSON data
  with open(file_path, 'r') as file:
    for index , json_row in enumerate(tqdm(file)):
      data = json.loads(json_row)
      if "image" in data:
        data['image'] = ",".join(data['image'])

      if index > 1_000_000 :
        break

      all_reviews_df.append(pd.DataFrame({"overall" : data.get("overall",None),
                                          "verified" :data.get("verified",None),
                                          "reviewText" : data.get("reviewText",None)},index=[0]))
  return all_reviews_df

size = 50_000

for items in data_link_dict.items():
  !wget "{items[1]}"

  gc.collect()

  ## Unzipping
  gz_file_path = f'{items[0]}.json.gz'
  output_file_path = f'{items[0]}.json'
  gunzip_file(gz_file_path, output_file_path)

  #Loading Data
  loaded_data = load_json_file(output_file_path)

  loaded_data_df = pd.concat(random.sample(loaded_data,size))

  DEST_PATH = f"/content/drive/MyDrive/Natural Language Understanding/Assignment1/data/{items[0]}.csv"

  loaded_data_df.to_csv(DEST_PATH,index = False)

  os.remove(gz_file_path)
  os.remove(output_file_path)
  del loaded_data
  del loaded_data_df
