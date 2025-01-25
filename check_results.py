
class Box:
    def __init__ (self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_area(self):
        return (self.x2-self.x1)*(self.y2-self.y1)

    def bbx_iou(self,box=None):
        x1 = max(self.x1, box.x1)
        y1 = max(self.y1, box.y1)
        x2 = min(self.x2, box.x2)
        y2 = min(self.y2, box.y2)
        inter = (x2-x1)*(y2-y1)
        ar = self.get_area() + box.get_area()
        iou = inter/(ar - inter)
        return iou

def get_bbox(x1,y1,x2,y2):
    bb = Box()
    bb.x1,bb.y1,bb.x2,bb.y2 = x1,y1,x2,y2
    return bb