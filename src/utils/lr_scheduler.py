class LRScheduler:
    """Schedules the learning rate changes during the trining process.It implements linear, 
    exponential and custom learning rate scheduling.
    """    
    def __init__(self, steps, lr, min_lr=1e-6, **kwargs):     
        self.steps = steps
        self.steps.sort()
        self.steps = [0] + self.steps
        
        if isinstance(lr, list):
            try:
                assert len(self.steps) == len(lr)
            except AssertionError:
                raise ValueError("The length or steps and lr should be the same")
            self.lr = lr
        else:
            scheduling_mode = kwargs["mode"]
            if scheduling_mode == "linear":
                lr_step = kwargs["lr_step"]
                self.lr = [max(min_lr, lr-n*lr_step) for n in range(len(self.steps))] 
            if scheduling_mode == "exp":
                lr_step = kwargs["lr_step"]
                self.lr = [max(min_lr, lr * lr_step**n) for n in range(len(self.steps))]
    
    def update(self, step):
        for index, value in enumerate(self.steps[::-1]):
            if value <= step:
                return self.lr[len(self.steps) - 1 - index] 


if __name__ == "__main__":
    lr_scheduler = LRScheduler([10, 20, 40, 50], 0.1, mode="exp", lr_step=0.2)
    print(lr_scheduler.update(0))
    print(lr_scheduler.update(15))
    print(lr_scheduler.update(45))
    print(lr_scheduler.update(55))