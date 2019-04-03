CCOMP = pgcc
CFLAGS = -acc -Minfo=acc -ta=tesla:cc70 -fast

run: vector_addition.o
	$(CCOMP) $(CFLAGS) vector_addition.o -o run

vector_addition.o: vector_addition.c
	$(CCOMP) $(CFLAGS) -c vector_addition.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o vec_add_cuda.* *.nvvp
