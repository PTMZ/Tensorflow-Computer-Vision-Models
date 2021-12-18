import tensorflow as tf 
from tensorflow.keras import layers, losses
from tensorflow.keras import Model
import tensorflow_hub as hub
from swintransformer import SwinTransformer

model_handle_dict = {
    "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
    "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
    "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
    "vit-b8": "https://tfhub.dev/sayakpaul/vit_b8_fe/1",
    "vit-b16": "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
}


class EffnetSwin(Model):
    def __init__(self, img_size):
        super(EffnetSwin, self).__init__()
        self.img_shape = (img_size, img_size, 3)
        self.swin = create_swin(compileloss=False, finetune=True, featextract=True)
        self.effnet = create_effnetv2(img_size=img_size, finetune=False, featextract=True, compileloss=False)
        self.fc1 = layers.Dense(75)
        self.fc2 = layers.Dense(75)
        self.warmup = 1
    
    def set_warmup(self, w, lr=1e-3):
        self.warmup = w
        self.recompile(lr=lr)

    def recompile(self, lr=1e-3):
        self.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
            loss = losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01),
            #loss = losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['acc']) 

    # calling 
    def call(self, inputs, training=None, **kwargs):
        if self.warmup == 1:
            return self.fc1(self.swin(inputs))
        if self.warmup == 2:
            return self.fc2(self.effnet(inputs))
        return self.fc1(self.swin(inputs)) + self.fc2(self.effnet(inputs))

    def build_graph(self):
        x = layers.Input(shape=self.img_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def summary(self):
        x = layers.Input(shape=self.img_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()    


def create_effnet_swin(img_size=480, print_summary=False, with_compile=False):
    model = EffnetSwin(img_size)
    if print_summary:
        print(model.summary())
    if with_compile:
        model.recompile()
    return model


def create_swin(img_size = 480, num_classes=75, compileloss=True, finetune=False, featextract=False):

    base_model = SwinTransformer('swin_base_384', include_top=False, pretrained=True)
    #base_model.basic_layers.trainable = False
    #base_model.norm.trainable = False
    if not finetune:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers:
            if layer not in base_model.layers[2:]:
                layer.trainable = False
        
        for layer in base_model.layers[2].layers:
            if layer not in base_model.layers[2].layers[3:]:
                layer.trainable = False
    
    # base_model.summary()
    

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = layers.experimental.preprocessing.Resizing(384, 384)(inputs)
    x = layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[384, 384, 3])(x)
    x = base_model(x)
    if featextract:
        output = x
    else:
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes)(x)
    model = Model(inputs, output)

    if compileloss:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01),
                    #loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
                    metrics=['acc'])
        
    
    return model


def create_effnetv2(img_size = 384, num_classes=75, model_size="s", finetune=False, featextract=False, compileloss=True):
    model_handle = model_handle_dict.get(f"efficientnetv2-{model_size}")
    img_size_dict = {"s":384, "m":480, "l":480}
    img_input_size = img_size_dict[model_size]


    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = layers.experimental.preprocessing.Resizing(img_input_size, img_input_size)(inputs)
    x = layers.Rescaling(1./255)(x)
    x = hub.KerasLayer(model_handle, trainable=finetune)(x)
    if featextract:
        output = x
    else:
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes)(x)
    model = Model(inputs, output)

    if compileloss:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                        loss=losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01),
                        # loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
                        metrics=['acc'])
    
    return model


def create_vit(img_size = 224, num_classes=75, model_name="vit-b8"):
    model_handle = model_handle_dict.get(model_name)

    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(img_size, img_size,3)),
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.Rescaling(1./127.5, offset=-1),
        hub.KerasLayer(model_handle, trainable=False),
        layers.Dropout(0.2),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                        loss=losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01), 
                        metrics=['acc'])
    
    return model


# Not used. Failed due to OOM.
# Original Hybrid Transofrmer found on Kaggle.
#
class EfficientHybridSwinTransformer(Model):
    def __init__(self, img_size):
        super(EfficientHybridSwinTransformer, self).__init__()
        self.img_shape = (img_size, img_size, 3)
        # base models 
        self.model_input_size = 448
        self.model_input_shape = (self.model_input_size, self.model_input_size, 3)
        self.resize_layer = layers.experimental.preprocessing.Resizing(self.model_input_size, self.model_input_size)
        self.resize_warmup = layers.experimental.preprocessing.Resizing(self.model_input_size//2, self.model_input_size//2)
        self.rescale = x = layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[224, 224, 3])
        self.inputx = layers.Input(self.model_input_shape, name='input_hybrids')
        self.warmup = True
        base = tf.keras.applications.EfficientNetB0(
            include_top  = False,
            weights      = None,
            input_tensor = self.inputx
        )
        for layer in base.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        # base model with compatible output which will be an input of transformer model 
        self.new_base = Model(
            base.inputs, 
            base.get_layer('block1a_project_bn').output, # output with 224 feat_maps
            name='efficientnet'
        )
        self.conv = layers.Conv2D(3, 3, padding='same')
        self.swin_blocks = SwinTransformer('swin_base_224', include_top=False, pretrained=True)
        self.swin_blocks.trainable = False
        self.fc = layers.Dense(75)

    def freeze(self):
        self.warmup = True
        self.swin_blocks.trainable = False
        self.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), 
            loss = losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01),
            metrics = ['acc'])  

    def unfreeze(self):
        self.warmup = False
        self.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4), 
            loss = losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01),
            metrics = ['acc'])  


    # calling 
    def call(self, inputs, training=None, **kwargs):
        if self.warmup:
            x = self.resize_warmup(inputs)
            x = self.rescale(x)
            return self.fc(self.swin_blocks(x))
        
        x = self.resize_layer(inputs)
        x = self.rescale(x)
        base_x = self.new_base(x)
        from_swin = self.swin_blocks(self.conv(base_x))
        return self.fc(from_swin)
        # cating = tf.concat([from_swin, layers.GlobalAveragePooling2D()(base_y)], axis=-1)
        # if training:
        #     return self.fc(cating)
        # else:
        #     return self.fc(cating), cating

    def build_graph(self):
        x = layers.Input(shape=self.img_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def summary(self):
        x = layers.Input(shape=self.img_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


def create_hybrid_swin(img_size=480, print_summary=False, with_compile=False):
    model = EfficientHybridSwinTransformer(img_size)

    if print_summary:
        #print(model(tf.ones((1, img_size, img_size, 3)))[0].shape)
        print(model.summary())
        
    if with_compile:
        #print(model(tf.ones((1, img_size, img_size, 3)))[0].shape)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4), 
            loss = losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.01),
            metrics = ['acc'])  
        
    return model 
