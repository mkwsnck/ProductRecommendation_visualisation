from dataclasses import dataclass
from functools import cached_property
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import numpy as np
import pyspark.sql.functions as fn
import requests
from PIL import Image
from pyspark.sql import DataFrame as SparkDataFrame

TOP = 15  # selected recommendations

@dataclass
class Product2ProductViewer:
    all_images_df: SparkDataFrame
    limit: int
    columns: int = TOP +1
    num_threads: int = 4
    label_font_size: int = 10

    @cached_property
    def images_df(self):
        return self.all_images_df.limit(self.limit).cache()

    @cached_property
    def cnt(self):
        return self.images_df.count()

    @cached_property
    def collected(self):
        return self.images_df.withColumn('size', fn.size(fn.col('supprodId2'))).collect()

    def get_single_image(self, image):
        prefix = 'https://hiddenlink'
        suffix = '/1.jpg'
        try:
            stream = requests.get(f"{prefix}{image}{suffix}", stream=True).raw
            img_content = Image.open(stream)
            return {image :img_content}
        except:
            pass

    def download_all(self, enumerated_row):
        row = enumerated_row[1]
        main = [self.get_single_image(row["supprodId"])]
        similar = [self.get_single_image(i) for i in row["supprodId2"]]
        return {row["supprodId"] : similar+main}

    @cached_property
    def full_dict(self):
        with ThreadPool(self.num_threads) as pool1:
            row_dict = pool1.map(self.download_all, enumerate(self.collected), chunksize=10)
        return dict((key, value) for element in row_dict for nested_dict in element.values() for item in nested_dict for key, value in item.items())


    def main_chart(self, enumerated_row):
        i = enumerated_row[0]
        row = enumerated_row[1]
        try:
            plt.subplot(self.cnt, self.columns, i* self.columns + 1)
            plt.imshow(self.full_dict[row["supprodId"]])
            plt.title(f"{row['size']}:{row['supprodId']}", fontsize=self.label_font_size + 1, color='red');
        except:
            pass
        else:
            self.single_chart(i, row)

    def single_chart(self, i, row):
        for col in range(0, self.columns - 1):
            try:
                plt.subplot(self.cnt, self.columns, i * self.columns + col + 2)
                plt.imshow(self.full_dict[row["supprodId2"][col]])
                if row["type"][col] == 5:  # p2p
                    plt.title(row["points_raw"][col], fontsize=self.label_font_size, color='green');
                elif row["type"][col] == 4:  # middle
                    plt.title(row["points_raw"][col], fontsize=self.label_font_size, color='blue');
                elif row["type"][col] == 2:  # 2 lvl
                    plt.title(row["points_raw"][col], fontsize=self.label_font_size, color='orange');
                elif row["type"][col] == 1:  # 3 lvl
                    plt.title(row["points_raw"][col], fontsize=self.label_font_size, color='red');
            except:
                empty_img = np.array([[[0, 1, 1, 1]]], dtype='uint8')
                plt.subplot(self.cnt, self.columns, i * self.columns + col + 2)
                plt.imshow(empty_img, interpolation='nearest')

    def plot(self):
        height = 1.6 * self.cnt
        width = 20

        plt.figure(figsize=(width, height))
        [self.main_chart(i) for i in enumerate(self.collected)]
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);