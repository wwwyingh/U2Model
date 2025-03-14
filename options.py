import argparse

def get_options():
    parser = argparse.ArgumentParser(description='ComplexProject Options')

    # Device and I/O settings
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    # Model architecture parameters
    parser.add_argument('--in_channel', type=int, default=1, help='Input channel count')
    parser.add_argument('--first_out_channel', type=int, default=64, help='Initial number of channels')
    parser.add_argument('--dropout_prob', type=float, default=0.3, help='Dropout probability')

    # Registration network parameters
    parser.add_argument('--flow_multiplier', type=float, default=1.0, help='Flow multiplier')
    parser.add_argument('--channels', type=int, default=16, help='Channel count for registration network')
    parser.add_argument('--registration_steps', type=int, default=7, help='Number of integration steps in VecInt')
    parser.add_argument('--image_shape', type=int, nargs=2, default=[192, 192], help='Image shape (height width)')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--epochs', type=int, default=401, help='Number of training epochs')
    parser.add_argument('--root_folder', type=str, default='/home/mpadmin/BraTs/', help='Data root folder')
    parser.add_argument('--save_dir', type=str, default='./save_our/', help='Directory to save models and logs')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loader')

    # Data augmentation parameters
    parser.add_argument('--max_displacement', type=int, default=120, help='Max displacement for elastic deformation')
    parser.add_argument('--num_control_points', type=int, default=100, help='Number of control points for elastic deformation')
    parser.add_argument('--sigma', type=int, default=20, help='Sigma value for elastic deformation')

    # Stochastic intensity transformation parameter
    parser.add_argument('--delta', type=float, default=0.5, help='Delta for intensity transformation')

    # Loss weight parameters
    parser.add_argument('--weight_smooth', type=float, default=1.0, help='Weight for smooth loss')
    parser.add_argument('--weight_fake_recon', type=float, default=10.0, help='Weight for fake reconstruction loss')
    parser.add_argument('--weight_recon', type=float, default=1.0, help='Weight for reconstruction loss')
    parser.add_argument('--weight_nmi', type=float, default=1.0, help='Weight for mutual information loss (A2B+B2A)')

    opts = parser.parse_args()
    opts.image_shape = tuple(opts.image_shape)
    return opts

if __name__ == "__main__":
    options = get_options()
    print(options)
