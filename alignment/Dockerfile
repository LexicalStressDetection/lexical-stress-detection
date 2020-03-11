FROM lowerquality/gentle:latest

RUN pip install numpy==1.13.0
RUN pip install scipy==0.16

COPY cmudict-0.7b.txt /gentle/
COPY gentle_alignment.py /gentle/

WORKDIR /gentle

CMD ["python", "-u", "gentle_alignment.py"]
