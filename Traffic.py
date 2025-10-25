from tkinter import Toplevel
from tkinter.ttk import Label, Progressbar, Style
from PIL import Image, ImageTk
import time
import pandas as pd
import json
import herepy

width = 400
height = 250

# Get latitude and longitude using HERE API
def location(place):
    geocoderApi = herepy.GeocoderApi('YOUR_HERE_API_KEY')  # Replace with your actual key
    response = geocoderApi.free_form(place)
    s = json.loads(str(response))
    lat = s["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]["Latitude"]
    lng = s["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]["Longitude"]
    lat = "{0:.2f}".format(lat)
    lng = "{0:.2f}".format(lng)
    return '{' + lat + ',' + lng + '}'

# Dictionary with locations
d = {
    1: "Big Ben " + location("Big Ben"),
    2: "Gariahat " + location("Gariahat"),
    3: "Jadavpur " + location("Jadavpur"),
    4: "Times Square " + location("Times Square"),
    5: "Rasbehari " + location("Rasbehari"),
    6: "Garia " + location("Garia"),
    7: "Tollygunge " + location("Tollygunge"),
    8: "Chingrihata " + location("Chingrihata"),
    9: "Saltlake " + location("Salt Lake"),
}


class Traffic(Toplevel):
    def __init__(self):
        super().__init__()
        self.title("TRAFFIC MANAGEMENT SYSTEM")
        self.configure(background="white")
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))

        for i in range(1, 3):  # Two sets of images
            for j in range(1, 10):  # 9 locations
                self.initUI(i, j, (j - 1) % 3 + 1, (j - 1) // 3 + 1)
            time.sleep(5)

    def initUI(self, path, pic, xi, yi):
        # Load image
        stgImg = Image.open(f"{(path - 1) * 5}//{pic}.jpg")
        stgImg = stgImg.resize((width, height), Image.ANTIALIAS)
        stgImg2 = ImageTk.PhotoImage(stgImg)

        # Load traffic data
        data = pd.read_csv(f"output{(path - 1) * 5}.csv", header=None)
        var = data.to_numpy()

        # Display image
        label = Label(self, image=stgImg2)
        label.image = stgImg2
        label.place(x=543 * xi - 523, y=280 * yi - 290)

        # Display progress bar and text
        label1 = Label(self)
        s = Style()
        s.theme_use('clam')
        s.configure("red.Horizontal.TProgressbar", foreground='red', background='red')

        progress = Progressbar(label1, style="red.Horizontal.TProgressbar", length=100, mode='determinate')
        progress['value'] = int(var[pic - 1][1] * 20)

        # Time in minutes and location name
        time_text = str(var[pic - 1][1]) + " mins"
        location_name = d[pic]

        label2 = Label(self, text=time_text, font="arial 12 bold", background="#f0d630")
        label2.place(x=543 * xi - 407, y=280 * yi - 33)

        label3 = Label(self, text=location_name, font="arial 12 bold", background="white")
        label3.place(x=543 * xi - 325, y=280 * yi - 32)

        progress.pack()
        label1.place(x=543 * xi - 520, y=280 * yi - 32)
        self.update_idletasks()
