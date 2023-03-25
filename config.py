import argparse

def config_data():
    parser = argparse.ArgumentParser(description="sentiment analysis")
    parser.add_argument(
        "--epoch", type=int, default=30,
        help="epoch"
    )
    parser.add_argument(
        "--batch_size", type=int, default=48,
        help="batch_size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="learning_rate"
    )
    parser.add_argument(
        "--cuda", type=int, default=1,
        help="cuda"
    )
    parser.add_argument(
        "--result_path", type=str, default="test",
        help="cuda"
    )
    
    parser.add_argument(
        "--latent_size", type=int, default=768,
        help="latent size"
    )
    
    parser.add_argument(
        "--contrastive", default=False, action='store_true',
        help="contrastive"
    )
    parser.add_argument(
        "--cls_latent_vector", default=False, action='store_true',
        help="cls_latent_vector"
    )
    
    
    parser.add_argument(
        "--dd", type=float, default=0.6,
        help="decoder loss weight"
    )
    parser.add_argument(
        "--dc", type=float, default=0.2,
        help="contrastive loss weight"
    )
    parser.add_argument(
        "--di", type=float, default=0.2,
        help="image regularization loss weight"
    )
    
    args = parser.parse_args()
    return args