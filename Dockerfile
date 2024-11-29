#FROM innovanon/ia_communicate AS communicate
#FROM innovanon/ia_syslog      AS syslog
#FROM innovanon/ia_sisyphus    AS sisyphus
#FROM innovanon/ia_spydir      AS spydir
FROM innovanon/ia_setup       AS setup

#COPY --from=communicate /tmp/py/ /tmp/py/
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#RUN pip install --no-cache-dir --upgrade .
#RUN rm -rf /tmp/py/

COPY ./ ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir --upgrade .
ENTRYPOINT ["python", "-m", "ia_ytshackleford"]
