from options import Options
from lib.data.dataloader import load_data_FD_aug
from lib.models import load_model
##
def train(opt,class_name):
    data = load_data_FD_aug(opt, class_name)
    model = load_model(opt, data, class_name)
    auc = model.test(load_weights=True)
    model.save_scores()
    return auc
def main():
    """ Training
    """
    opt = Options().parse()
    train(opt, opt.dataset)
if __name__ == '__main__':
    main()
