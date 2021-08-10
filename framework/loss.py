class Loss(object):
    pass

class MSELoss(object):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def __call__(self, pred, gt):
        diff = pred - gt
        sqdiff = diff * diff
        loss = sqdiff.sum(0)
        return loss


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, target):
        '''
        The function combines softmax & cross entropy
        '''
        return pred.cross_entropy(target)