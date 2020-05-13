from torch import nn
from discriminator import Discriminator
from generator import Generator

class HoloGAN(nn.Module):
    def __init__(self, input_height=108, input_width=108 ,output_height=64, output_width=64,
                 crop=True, gf_dim=64, df_dim=64, c_dim=3,
                 dataset_name='lsun', input_fname_pattern='*.webp', **kwargs):
        super(HoloGAN, self).__init__()
        # TODO: bunlardan hangileri bize lazım olacak bilmiyorum, elemek gerekebilir
        self.crop = crop

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        # TODO: in feature we may need to transform the weigts of our layers to check the exact results
        self.discriminator = Discriminator(inplanes=self.c_dim, planes=df_dim, cont_dim=kwargs["z_dim"], reuse=False)

        # TODO: buradaki z değişkeni batch ile değişecek yani forwardda olmalı aslında
        self.generator = Generator(kwargs["z_dim"], self.gf_dim)

        """
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
        self.checkpoint_dir = LOGDIR
        """

    def forward(self, x):
        return self.discriminator(x)


    # TODO: def generator():


    # TODO: def train():

    # TODO: def sample():
