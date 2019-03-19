CCOMP = pgcc
CFLAGS = -Minfo -fast

run: poisson2d.o
	$(CCOMP) $(CFLAGS) poisson2d.o -o run

poisson2d.o: poisson2d.c
	$(CCOMP) $(CFLAGS) -c poisson2d.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f *.nvvp *.prof *.err *.out

