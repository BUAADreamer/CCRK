class TextObject:
    def __init__(self, lan, text_data):
        self.lan = lan
        self.text_ids, self.text_atts, self.text_ids_masked, self.masked_pos, self.masked_ids = text_data
