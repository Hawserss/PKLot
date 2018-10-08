class Result:
    def __init__(self, date):
        self.trueNegative = 0
        self.falseNegative = 0
        self.truePositive = 0
        self.falsePositive = 0
        self.date = date
    
    def setValue(self, value, label):
        if label == 0:
            if value == label:                
                self.trueNegative += 1
            else:
                self.falseNegative += 1
        else:
            if value == label:
                self.truePositive += 1
            else:
                self.falsePositive += 1

    def getTA(self):
        trues = (self.truePositive + self.trueNegative)
        totals = (self.truePositive + self.falsePositive) + (self.trueNegative + self.falseNegative)
        return trues/totals