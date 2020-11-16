%.out: %.in
	/Users/will/Work/CLOUDY/cloudy/source/cloudy.exe -p $* 

OUTFILES := $(patsubst %.in,%.out,$(wildcard trap-shock*.in))

all: $(OUTFILES)

