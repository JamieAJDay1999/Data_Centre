"""Vector redraw of the existing facility schematic, preserving its topology."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE=Path(__file__).resolve().parent
OUT=HERE/'images/Figure_1.pdf'
font_path=Path('C:/Windows/Fonts/arial.ttf')
if not font_path.exists():
    from matplotlib.font_manager import findfont
    font_path=Path(findfont('DejaVu Sans'))
pdfmetrics.registerFont(TTFont('Schematic',str(font_path)))
# Start with the embedded font so the PDF never references unembedded Helvetica.
c=canvas.Canvas(str(OUT),pagesize=(1141,831),initialFontName='Schematic')
c.setTitle('Data centre layout')
def line(points,colour='black',width=1):
    c.setStrokeColor(HexColor(colour) if colour.startswith('#') else colour)
    c.setLineWidth(width)
    path=c.beginPath();path.moveTo(points[0][0],831-points[0][1])
    for x,y in points[1:]:path.lineTo(x,831-y)
    c.drawPath(path)
def arrow(points,colour='black',width=1):
    import math
    line(points,colour,width)
    x,y=points[-1];px,py=points[-2];a=math.atan2(y-py,x-px)
    length=8;width=3.5
    tri=[(x,y),(x-length*math.cos(a)+width*math.sin(a),y-length*math.sin(a)-width*math.cos(a)),
         (x-length*math.cos(a)-width*math.sin(a),y-length*math.sin(a)+width*math.cos(a)),(x,y)]
    c.setFillColor(HexColor(colour) if colour.startswith('#') else colour)
    p=c.beginPath();p.moveTo(tri[0][0],831-tri[0][1])
    for x,y in tri[1:]:p.lineTo(x,831-y)
    c.drawPath(p,fill=1,stroke=0)
def text(x,y,value,size=24):
    c.setFillColor('black');c.setFont('Schematic',size)
    rows=value.split('\n')
    for k,row in enumerate(rows):c.drawCentredString(x,831-y-k*(size+4),row)
def box(x,y,w,h,value='',size=24):
    c.setStrokeColor('black');c.setFillColor('white');c.setLineWidth(1)
    c.rect(x,831-y-h,w,h,fill=1)
    if value:text(x+w/2,y+h/2-(len(value.split('\n'))-1)*(size+4)/2+size/3,value,size)
def tag(x,y,label):
    c.setFillColor(HexColor('#AEF76C'));c.rect(x,831-y-30,30,30,stroke=0,fill=1)
    text(x+15,y+22,label,19)
box(430,0,380,475)
line([(620,0),(620,475)])
text(525,60,'Air in Cold\nAisle')
text(715,60,'Air in Hot\nAisle')
box(544,125,136,175);text(612,164,'Air in Rack')
box(579,180,67,100);text(612,222,'ITE')
box(220,140,120,101,'UPS\nBattery')
box(0,335,131,101,'Grid\nConnection')
box(220,335,120,101,'Power\nDelivery\nSystem')
box(220,560,120,101,'Chiller')
box(460,560,120,101,'HVAC')
box(220,720,120,101,'TES Tank')
arrow([(131,382),(220,382)])
arrow([(280,335),(280,241)])
arrow([(280,436),(280,560)])
arrow([(340,191),(579,191)])
arrow([(340,620),(460,620)],'#0000FF',2)
arrow([(460,640),(340,640)],'#FF4040',2)
arrow([(280,661),(280,720)],'#0000FF',2)
arrow([(340,760),(520,760),(520,661)],'#0000FF',2)
arrow([(520,560),(520,475)],'#0000FF',.8)
arrow([(810,115),(879,115),(879,510),(550,510),(550,560)],'#FF4040',.8)
for x in [530,569]:arrow([(x,230),(x+34,230)],'#0000FF',.8)
for x in [630,669]:arrow([(x,230),(x+34,230)],'#FF4040',.8)
for x,y,t in [(482,490,'T1'),(500,368,'T2'),(649,176,'T3'),(596,241,'T4'),(703,369,'T5'),(888,271,'T5')]:tag(x,y,t)
for y,t,label in [(531,'T1','Air inlet temp'),(590,'T2','Cold aisle temp'),(651,'T3','Rack temp'),(711,'T4','ITE temp'),(771,'T5','HVAC return / Hot\naisle temp')]:
    tag(639,y,t);text(782,y+25,label,23)
for y,label,col,width in [(545,'Power','black',1),(601,'Hot Water','#FF4040',2),(670,'Cool\nWater','#0000FF',2),(741,'Hot Air','#FF4040',.8),(800,'Cool Air','#0000FF',.8)]:
    arrow([(920,y),(1000,y)],col,width);text(1080,y+7,label,23)
c.showPage();c.save()
print(OUT)
