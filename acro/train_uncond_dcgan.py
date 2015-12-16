"""
uncond_dcgan1 made with 64x64 images from https://s3.amazonaws.com/udipublic/acro.images.tgz for train.tar.gz
"""
import argparse

parser = argparse.ArgumentParser(description='train uncoditional dcgan')
parser.add_argument('--desc',
                    default='uncond_dcgan',
                    help='name to uniquely describe this run')
parser.add_argument('--path',
                    default='data/jpg.hdf5',
                   help='where to read fuel hdf5 data file with training')
parser.add_argument('--val', type=float,
                    default=0.,
                   help="what part of the training data to use for validation")
parser.add_argument('--model',
                   help='start from a pre-existing model.'
                        ' The suffixes _gen_params.jl'
                        ' and _discrim_params.jl'
                        ' are added to the path you supply')
parser.add_argument('--batch', type=int,
                    default=128,
                   help='batch size')
parser.add_argument('-k', type=int,
                    default=0,
                   help='# of discrim updates for each gen update.'
                        ' 0 - alternate > 0 more d, < 0 more g')
parser.add_argument('--maxk', type=int,
                    default=1,
                   help='max value for k')
parser.add_argument('--mink', type=int,
                    default=-1,
                   help='min value for k')
parser.add_argument('--l2d', type=float,
                    default=1.e-5,
                   help="discriminator l2")
parser.add_argument('--l2decay', type=float,
                    default=0.,
                   help="reduce l2d by 1-l2decay")
parser.add_argument('--l2step', type=float,
                    default=0.,
                   help="increase(decrease) discriminator's l2"
                        " when generator cost is above 1.3(below 0.9)")
parser.add_argument('--dropout', type=float,
                    default=0.,
                   help="discriminator dropout")
parser.add_argument('--lr', type=float,
                    default=0.0002,
                   help="initial learning rate for adam")
parser.add_argument('--lrstep', type=float,
                    default=1.,
                   help="increa/decrease g/d learning rate")
parser.add_argument('--dbn', action='store_false',
                    help='dont perfrom batch normalization on discriminator')
parser.add_argument('--db1', action='store_true',
                    help='add bias to first layer of discriminator')
parser.add_argument('--ngf', type=int,
                    default=128,
                   help='# of gen filters')
parser.add_argument('--ndf', type=int,
                    default=128,
                   help='# of discriminator filters')
parser.add_argument('--updates', type=int,
                    default=100,
                   help='compute score every n_updates')
parser.add_argument('-z', type=int,
                    default=100,
                   help='number of hidden variables')
parser.add_argument('--znorm', action='store_true',
                    help='normalize z values to unit sphere')
parser.add_argument('--generate', action='store_true',
                    help='generate sample png and gif')
parser.add_argument('--ngif', type=int, default=1,
                   help='# of png images to generate. If 1 then no gif')
parser.add_argument('--nvis2', type=int,
                    default=14,
                   help='number of rows/cols of sub-images to generate')
parser.add_argument('--generate_d', type=float, default=0.,
                   help="minimal discrimation score when generating samples")
parser.add_argument('--generate_c', type=float, default=0.,
                   help="minimal classification score when generating samples")
parser.add_argument('--generate_v', type=float,
                    help='generate sample along a random direction with this step size')
parser.add_argument('--classify', action='store_true',
                    help='classify target')
parser.add_argument('--onlyclassify', action='store_true',
                    help='just do classify target')
parser.add_argument('--seed', type=int,
                    default=123,
                   help='seed all random generators')
parser.add_argument('--filter_label', type=int,
                   help='take only training data with this label (does not work with classify')
parser.add_argument('--nepochs', type=int,
                    default=25,
                   help='total number of epochs')
parser.add_argument('--niter', type=int,
                    default=25,
                   help='# of iter at starting learning rate')
parser.add_argument('--start', type=int,
                    default=0,
                   help='If not 0 then start from this epoch after loading the last model')
args = parser.parse_args()
if args.onlyclassify:
    args.classify = True
if args.classify:
    assert args.filter_label is None, "you can't classify and limit your data to one lable"
if args.model is None and args.start > 0:
    args.model = 'models/%s/%d'%(args.desc, args.start)


import random
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)


import sys
sys.path.append('..')

import os
import json
from time import sleep
from time import time
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from load import streams

def transform(X):
    # X = [center_crop(x, npx) for x in X]  # only works for (H,W,3)
    assert X[0].shape == (npx,npx,3) or X[0].shape == (3,npx,npx)
    if X[0].shape == (npx,npx,3):
        X = X.transpose(0, 3, 1, 2)
    return floatX(X/127.5 - 1.)

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X

k = 0             # # of discrim updates for each gen update. 0 - alternate > 0 more d, < 0 more g
l2 = 1e-5         # l2 weight decay
l2d = args.l2d     # discriminator l2
l2step = args.l2step     # increase(decrease) discriminator l2 when generator cost is above 1.3(below 0.9)
margin = 0.3    # Dont optimize discriminator(generator) when classification error below margin(above 1-margin)
nvis2 = args.nvis2
nvis = nvis2*nvis2        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = args.batch      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = args.z          # # of dim for Z
ngf = args.ngf         # # of gen filters in first conv layer
ndf = args.ndf         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = args.niter  # # of iter at starting learning rate
niter_decay = args.nepochs - niter   # # of iter to linearly decay learning rate to zero
lr = args.lr       # initial learning rate for adam
ntrain = None   # # of examples to train on. None take all
ngif = args.ngif  # # of images in a gif

desc = args.desc
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

###########################################
# data
if not args.generate:
    tr_data, tr_stream, val_stream, ntrain_s, nval_s = streams(ntrain=ntrain,
                                             batch_size=args.batch,
                                             path=args.path,
                                             val = args.val,
                                             filter_label=args.filter_label)
    if ntrain is None:
        ntrain = tr_data.num_examples
    print '# examples', tr_data.num_examples
    print '# training examples', ntrain_s
    print '# validation examples', nval_s

    tr_handle = tr_data.open()
    vaX,labels = tr_data.get_data(tr_handle, slice(0, 10000))
    vaX = transform(vaX)
    means = labels.mean(axis=0)
    print('labels ',labels.shape,means,means[0]/means[1])

    vaY,labels = tr_data.get_data(tr_handle, slice(10000, min(ntrain, 20000)))
    vaY = transform(vaY)

    va_nnd_1k = nnd_score(vaY.reshape((len(vaY),-1)), vaX.reshape((len(vaX),-1)), metric='euclidean')
    print 'va_nnd_1k = %.2f'%(va_nnd_1k)
    means = labels.mean(axis=0)
    print('labels ',labels.shape,means,means[0]/means[1])

#####################################
# shared variables
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

gw  = gifn((nz, ngf*8*4*4), 'gw')
gg = gain_ifn((ngf*8*4*4), 'gg')
gb = bias_ifn((ngf*8*4*4), 'gb')
gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2')
gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2')
gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3')
gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3')
gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4')
gg4 = gain_ifn((ngf), 'gg4')
gb4 = bias_ifn((ngf), 'gb4')
gwx = gifn((ngf, nc, 5, 5), 'gwx')

dw  = difn((ndf, nc, 5, 5), 'dw')
db = bias_ifn((ndf), 'db')
dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2')
dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3')
dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3')
dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4')
db4 = bias_ifn((ndf*8), 'db4')
dwy = difn((ndf*8*4*4, 1), 'dwy')
dwy1 = difn((ndf*8*4*4, 1), 'dwy')

# models
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

# generator model
gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x

# discriminator model
"""
#old model
if args.dbn:
    if args.db1:
        print "Bias on layer 1 + batch normalization"
        discrim_params = [dw, db, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy, dwy1]
        def discrim(X, w, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy, wy1):
            h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2))+b.dimshuffle('x', 0, 'x', 'x'))
            h = dropout(h, args.dropout)
            h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
            h2 = dropout(h2, args.dropout)
            h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
            h3 = dropout(h3, args.dropout)
            h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
            h4 = dropout(h4, args.dropout)
            h4 = T.flatten(h4, 2)
            y = sigmoid(T.dot(h4, wy))
            y1 = sigmoid(T.dot(h4, wy1))
            return y, y1
    else:
        print "Batch normalization"
        discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy, dwy1]
        def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy, wy1):
            h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
            h = dropout(h, args.dropout)
            h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
            h2 = dropout(h2, args.dropout)
            h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
            h3 = dropout(h3, args.dropout)
            h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
            h4 = dropout(h4, args.dropout)
            h4 = T.flatten(h4, 2)
            y = sigmoid(T.dot(h4, wy))
            y1 = sigmoid(T.dot(h4, wy1))
            return y, y1
else:
    if args.db1:
        print "Bias on layer 1"
        discrim_params = [dw, db, dw2, db2, dw3, db3, dw4, db4, dwy, dwy1]
        def discrim(X, w, b, w2, b2, w3, b3, w4, b4, wy, wy1):
            h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2))+b.dimshuffle('x', 0, 'x', 'x'))
            h = dropout(h, args.dropout)
            h2 = lrelu(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))+b2.dimshuffle('x', 0, 'x', 'x'))
            h2 = dropout(h2, args.dropout)
            h3 = lrelu(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2))+b3.dimshuffle('x', 0, 'x', 'x'))
            h3 = dropout(h3, args.dropout)
            h4 = lrelu(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2))+b4.dimshuffle('x', 0, 'x', 'x'))
            h4 = dropout(h4, args.dropout)
            h4 = T.flatten(h4, 2)
            y = sigmoid(T.dot(h4, wy))
            y1 = sigmoid(T.dot(h4, wy1))
            return y, y1
    else:
        discrim_params = [dw, dw2, db2, dw3, db3, dw4, db4, dwy, dwy1]
        def discrim(X, w, w2, b2, w3, b3, w4, b4, wy, wy1):
            h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
            h = dropout(h, args.dropout)
            h2 = lrelu(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))+b2.dimshuffle('x', 0, 'x', 'x'))
            h2 = dropout(h2, args.dropout)
            h3 = lrelu(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2))+b3.dimshuffle('x', 0, 'x', 'x'))
            h3 = dropout(h3, args.dropout)
            h4 = lrelu(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2))+b4.dimshuffle('x', 0, 'x', 'x'))
            h4 = dropout(h4, args.dropout)
            h4 = T.flatten(h4, 2)
            y = sigmoid(T.dot(h4, wy))
            y1 = sigmoid(T.dot(h4, wy1))
            return y, y1
"""
#new model
discrim_params = [dw, db, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy, dwy1]
def discrim(X, w, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy, wy1):
    h0 = dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2))
    if args.db1:
        h0 += b.dimshuffle('x', 0, 'x', 'x')
    h1 = lrelu(h0)
    h1 = dropout(h1, args.dropout)
    h1 = dnn_conv(h1, w2, subsample=(2, 2), border_mode=(2, 2))
    if args.dbn:
        h1 = batchnorm(h1, g=g2, b=b2)
    else:
        h1 += b2.dimshuffle('x', 0, 'x', 'x')
    h2 = lrelu(h1)
    h2 = dropout(h2, args.dropout)
    h2 = dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2))
    if args.dbn:
        h2 = batchnorm(h2, g=g3, b=b3)
    else:
        h2 += b3.dimshuffle('x', 0, 'x', 'x')
    h3 = lrelu(h2)
    h3 = dropout(h3, args.dropout)
    h3 = dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2))
    if args.dbn:
        h3 = batchnorm(h3, g=g4, b=b4)
    else:
        h3 += b4.dimshuffle('x', 0, 'x', 'x')
    h4 = lrelu(h3)
    h4 = dropout(h4, args.dropout)
    h4 = T.flatten(h4, 2)
    y = sigmoid(T.dot(h4, wy))
    y1 = sigmoid(T.dot(h4, wy1))
    return y, y1


X = T.tensor4()
Z = T.matrix()
Y = T.matrix()
MASK = T.matrix()

gX = gen(Z, *gen_params)
p_gen, p_gen_classify = discrim(gX, *discrim_params)
p_real, p_classify = discrim(X, *discrim_params)


if args.model is not None:
    print 'loading',args.model
    from itertools import izip
    gen_params_values = joblib.load(args.model + '_gen_params.jl')
    for p, v in izip(gen_params, gen_params_values):
        p.set_value(v)
    discrim_params_values = joblib.load(args.model + '_discrim_params.jl')
    if len(discrim_params) == len(discrim_params_values):
        load_params = discrim_params
    else:  # support old save format
        print 'loading old format',len(discrim_params),len(discrim_params_values)
        if args.dbn and args.db1:
            raise Exception('impossible')
            load_params = [dw, db, dw2,  dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy, dwy1]
        elif args.dbn:
            load_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy, dwy1]
        elif args.db1:
            load_params = [dw, db,  dw2, db2, dw3, db3, dw4, db4, dwy, dwy1]
        else:
            load_params = [dw, dw2, db2, dw3, db3, dw4, db4, dwy, dwy1]
    assert len(discrim_params_values) == len(load_params), "# params in model does not match"

    for p, v in izip(load_params, discrim_params_values):
        p.set_value(v)

###############################
# generate
_gen = theano.function([Z], gX)

from sklearn.preprocessing import normalize
def gen_z(n):
    if args.znorm:
        return floatX(normalize(np_rng.uniform(-1., 1., size=(n, nz))))
    else:
        return floatX(np_rng.uniform(-1., 1., size=(n, nz)))

if args.generate:
    _genscore = theano.function([Z], [gX, p_gen, p_gen_classify])
    t = iter(trange(nvis))
    pgs = []
    pcs = []
    zmbs = []
    samples = []
    while len(zmbs) < nvis:
        zmb = gen_z(args.batch)
        xmb, pg, pc = _genscore(zmb)
        pgs.append(pg)
        pcs.append(pc)
        for i in range(args.batch):
            if pg[i] >= args.generate_d and pc[i] >= args.generate_c:
                zmbs.append(zmb[i])
                samples.append(xmb[i])
                t.next()
                if len(zmbs) >= nvis:
                    break

    pgs = np.concatenate(pgs)
    pcs = np.concatenate(pcs)
    print 'generate_d',pgs.mean(),pgs.std(),'generate_c',pcs.mean(),pcs.std()
    samples = np.asarray(samples)
    color_grid_vis(inverse_transform(samples), (nvis2, nvis2),
                   '%s/Z_%03d.png'%(samples_dir,0))

    if args.generate_v is None:
        sample_zmb0 = np.array(zmbs)
        sample_zmb1 = np.roll(sample_zmb0, 1, axis=0)
        for i in tqdm(range(1,ngif)):
            z = abs(1.-2.*i/(ngif-1.)) # from 1 to 0 and back to almost 1
            sample_zmb = z * sample_zmb0 + (1-z) * sample_zmb1
            samples = np.asarray(_gen(sample_zmb))
            color_grid_vis(inverse_transform(samples), (nvis2, nvis2),
                           '%s/Z_%03d.png'%(samples_dir,i))
    else:
        sample_zmb = np.array(zmbs)
        v = gen_z(nvis)
        for i in tqdm(range(1,ngif)):
            sample_zmb += args.generate_v * v
            samples = np.asarray(_gen(sample_zmb))
            color_grid_vis(inverse_transform(samples), (nvis2, nvis2),
                           '%s/Z_%03d.png'%(samples_dir,i))
    if ngif > 1:
        os.system("convert -delay 15 -loop 0 {0}/Z_*.png {0}/Z.gif".format(samples_dir))
    exit(0)


def gen_samples(n, nbatch=128):
    samples = []
    n_gen = 0
    for i in range(n/nbatch):
        zmb = gen_z(nbatch)
        xmb = _gen(zmb)
        samples.append(xmb)
        n_gen += len(xmb)
    n_left = n-n_gen
    if n_left:
        zmb = gen_z(n_left)
        xmb = _gen(zmb)
        samples.append(xmb)
    return np.concatenate(samples, axis=0)

####################
d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_classify = (bce(p_classify, Y) * MASK).sum() / MASK.sum()
d_classify_error = (T.neq(p_classify > 0.5, Y) * MASK).sum() / MASK.sum()
d_error_real = 1.-T.mean(p_real)
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
d_error_gen = T.mean(p_gen)
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_real + d_cost_gen
if args.onlyclassify:
    d_cost = d_classify
elif args.classify:
    d_cost += d_classify
g_cost = g_cost_d

cost_target = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen, d_error_real, d_error_gen, d_classify, d_classify_error]
lrg = sharedX(lr)
lrd = sharedX(lr)
l2t = sharedX(l2d)
d_updater = updates.Adam(lr=lrd, b1=b1, regularizer=updates.Regularizer(l2=l2t))
g_updater = updates.Adam(lr=lrg, b1=b1, regularizer=updates.Regularizer(l2=l2))
"""
#old model
if args.onlyclassify:
    d_updates = d_updater(discrim_params[:-2]+discrim_params[-1:], d_cost)
elif args.classify:
    d_updates = d_updater(discrim_params, d_cost)
else:
    d_updates = d_updater(discrim_params[:-1], d_cost)
"""
#new model
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

_train_g = theano.function([X, Z, Y, MASK], cost_target, updates=g_updates)
_train_d = theano.function([X, Z, Y, MASK], cost_target, updates=d_updates)
if args.onlyclassify:
    _train_classify = theano.function([X, Y, MASK], [d_classify, d_classify_error], updates=d_updates)
if args.classify:
    _classify_d = theano.function([X, Y, MASK], [d_classify, d_classify_error])

log_fields = [
    'n_epochs',
    'n_updates',
    'n_examples',
    'n_seconds',
    '1k_va_nnd',
    # '10k_va_nnd',
    # '100k_va_nnd',
    'g_cost',
    'd_cost',
    'error_r',
    'error_g',
    'd_cost_real',
    'd_cost_gen',
    'd_classify',
    'd_classify_error',
    'lrg','lrd',
    'l2d',
]

n_updates = 0
n_epochs = 0
n_examples = 0
do_initial_valid = True

log_lines = []
if args.start > 0:
    f_log = open('logs/%s.ndjson'%desc, 'rb')
    for l in f_log:
        j = json.loads(l.strip())
        if 'valid_classify' in j:
            do_initial_valid = False
            continue
        if j['n_epochs'] > args.start:
            break

        do_initial_valid = True
        n_epochs = j['n_epochs']
        n_updates = j['n_updates']
        n_examples = j['n_examples']
        lrg.set_value(floatX(j['lrg']))
        lrd.set_value(floatX(j['lrd']))
        l2t.set_value(floatX(j['l2d']))

        log_lines.append(l)
    f_log.close()

f_log = open('logs/%s.ndjson'%desc, 'wb')
for l in log_lines:
    f_log.write(l)

vis_idxs = py_rng.sample(np.arange(len(vaX)), nvis)
vaX_vis = inverse_transform(vaX[vis_idxs])
color_grid_vis(vaX_vis, (args.nvis2, args.nvis2), 'samples/%s_etl_test.png'%desc)

sample_zmb = gen_z(nvis)

vaX = vaX.reshape(len(vaX), -1)

print desc.upper()
t = time()
costs = []
label_sums = np.zeros(2)

def validate():
    if args.classify and args.val > 0.:
        sleep(5.)
        valid_label_sums = np.zeros(2)
        val_costs = []
        for imb,labels in tqdm(val_stream.get_epoch_iterator(), total=nval_s/nbatch):
            valid_label_sums += labels.sum(axis=0)
            y = labels[:,0].reshape((-1,1))
            mask = labels[:,1].reshape((-1,1))
            imb = transform(imb)
            cost = _classify_d(imb, y, mask)
            val_costs.append(cost)

        print 'valid label sums',valid_label_sums,valid_label_sums[0]/(valid_label_sums[1]+1e-8)
        val_cost = np.array(val_costs).mean(axis=0)
        d_cost_class = float(val_cost[0])
        d_error_class = float(val_cost[1])
        print("val_d_classify=%f val_d_classify_error=%f"%(d_cost_class, d_error_class))
        log = [d_cost_class, d_error_class]
        f_log.write(json.dumps(dict(zip(['valid_classify', 'valid_classify_error'], log)))+'\n')
        f_log.flush()
        sleep(5.)

if do_initial_valid:
    validate()

for epoch in range(args.start,args.nepochs):
    for imb,labels in tqdm(tr_stream.get_epoch_iterator(), total=ntrain_s/nbatch):
        label_sums += labels.sum(axis=0)
        y = labels[:,0].reshape((-1,1))
        mask = labels[:,1].reshape((-1,1))
        imb = transform(imb)
        if args.onlyclassify:
            cost = _train_classify(imb, y, mask)
            cost = [0]*(len(cost_target)-len(cost)) + cost
        else:
            zmb = gen_z(len(imb))
            if k >= 0:
                if n_updates % (k+2) == 0:
                    cost = _train_g(imb, zmb, y, mask)
                else:
                    cost = _train_d(imb, zmb, y, mask)
            else:
                if n_updates % (-k+2) == 0:
                    cost = _train_d(imb, zmb, y, mask)
                else:
                    cost = _train_g(imb, zmb, y, mask)
        n_updates += 1
        n_examples += len(imb)
        costs.append(cost)

        if n_updates % args.updates == 0:
            cost = np.array(costs).mean(axis=0)
            # [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen, d_error_real, d_error_gen,d_classify, d_classify_error]
            print 'label sums',label_sums,label_sums[0]/(label_sums[1]+1e-8)
            label_sums = np.zeros(2)
            costs = []
            g_cost = float(cost[0])
            d_cost = float(cost[1])
            d_cost_real = float(cost[3])
            d_cost_gen = float(cost[4])
            d_error_r = float(cost[5])
            d_error_g = float(cost[6])
            d_cost_class = float(cost[7])
            d_error_class = float(cost[8])
            gX = gen_samples(10000)
            gX = gX.reshape(len(gX), -1)
            va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
            # va_nnd_10k = nnd_score(gX[:10000], vaX, metric='euclidean')
            # va_nnd_100k = nnd_score(gX[:100000], vaX, metric='euclidean')
            log = [n_epochs, n_updates, n_examples, time()-t,
                   va_nnd_1k, g_cost, d_cost,
                   d_error_r, d_error_g,d_cost_real,d_cost_gen,
                   d_cost_class, d_error_class,
                   float(lrg.get_value()),float(lrd.get_value()),float(l2t.get_value())
                   ]
            print '%d %d %.2f'%(epoch, n_updates, va_nnd_1k)
            print 'gc=%.4f dc=%.4f dcr=%.4f dcg=%.4f er=%.4f eg=%.4f cls=%.4f err=%.4f'%(
                g_cost, d_cost, d_cost_real, d_cost_gen,
                d_error_r,d_error_g, d_cost_class, d_error_class)
            f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
            f_log.flush()

            # if g_cost > d_cost + .3:
            #     k -= 1
            # elif g_cost < d_cost - .3:
            #     k += 1
            # k = max(-3, min(3,k))
            # k poistive is do more d, k negative is do more g
            if d_error_r < margin or d_error_g < margin:  # d is too good
                k += args.k
                lrg.set_value(floatX(lrg.get_value()*args.lrstep))
                lrd.set_value(floatX(lrd.get_value()/args.lrstep))
            elif d_error_r > 1.-margin or d_error_g > 1.-margin:  # d is too bad
                k -= args.k
                lrg.set_value(floatX(lrg.get_value()/args.lrstep))
                lrd.set_value(floatX(lrd.get_value()*args.lrstep))
            elif k > 0:  # unwind d
                k -= 1
                # lrd.set_value(floatX(lrd.get_value()/args.lrstep))
            elif k < 0:  # unwind g
                k += 1
                # lrg.set_value(floatX(lrg.get_value()/args.lrstep))
            k = max(args.mink,min(args.maxk,k))


            # http://torch.ch/blog/2015/11/13/gan.html#balancing-the-gan-game
            if g_cost > 1.3:  # g is bad -> increase regularization on d
                l2t.set_value(floatX(l2t.get_value() + l2step))
            elif g_cost < 0.9: # g is good -> decrease regularization on d
                l2t.set_value(floatX(l2t.get_value() - l2step))
            else:
                l2t.set_value(floatX(l2t.get_value() * (1.-args.l2decay)))
            if l2t.get_value() < 0:
                l2t.set_value(floatX(0.))
            print k, l2t.get_value()

    validate()

    samples = np.asarray(_gen(sample_zmb))
    color_grid_vis(inverse_transform(samples), (args.nvis2, args.nvis2), 'samples/%s/%d.png'%(desc, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrg.set_value(floatX(lrg.get_value() - lr/niter_decay))
        lrd.set_value(floatX(lrd.get_value() - lr/niter_decay))
    if n_epochs <= 5 or n_epochs % 5 == 0:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))
