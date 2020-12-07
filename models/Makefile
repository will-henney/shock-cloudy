%.out: %.in
	/Users/will/Work/CLOUDY/cloudy/source/cloudy.exe -p $* 

OUTFILES := $(patsubst %.in,%.out,$(wildcard trap-pshock*.in))

all: $(OUTFILES)

