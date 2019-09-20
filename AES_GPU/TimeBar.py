import time

class TimeBar(object):
    # Example:
    # timebar = TimeBar()
    # start = 371
    # for i in range(start + 1):
    #     print(timebar(i, start), end='')
    #     time.sleep(0.01)
    
    
    def __init__(self, number=100, decimal=2):
        self.decimal = decimal
        self.number = number
        self.a = 100/number

    def __call__(self, now, total):
        percentage = self.percentage_number(now, total)
        well_num = int(percentage / self.a)
        # print("well_num: ", well_num, percentage)
        progress_bar_num = self.progress_bar(well_num)
        result = "\r%s %s" % (progress_bar_num, percentage)
        return result

    def percentage_number(self, now, total):
        return round(now / total * 100, self.decimal)

    def progress_bar(self, num):
        well_num = "#" * num
        space_num = " " * (self.number - num)
        return '[%s%s]' % (well_num, space_num)