module driver();
  logic clock = 0;
  logic reset = 0;
  logic out0_valid, out0_ready;
  logic [63:0] out0_data;
  logic inCtrl_valid, inCtrl_ready;
  logic outCtrl_valid, outCtrl_ready;

  logic in0_valid, in0_ready;
  logic [63:0] in0_data;

  top dut (.*);

  always begin
    // A clock period is #4.
    clock = ~clock;
    #2;
  end

  logic wasRdy;
  logic [15:0] clkCnt = 0;

  // TODO write with functions
  initial begin
    out0_ready = 1;
    outCtrl_ready = 1;
    in0_valid = 0;
    inCtrl_valid = 0;

    reset = 1;
    // Hold reset high for one clock cycle.
    @(posedge clock);
    reset = 0;

    // give reset some time
    @(posedge clock);
    @(posedge clock);

    // wait until reset is done before we start to send data.

    // sending data
    in0_valid = 1;
    inCtrl_valid = 1;

    in0_data = 0;

    #0.1

    // TODO: wait individually
    // wait until the first transfer can happen
    wait(in0_ready == 1 && inCtrl_ready == 1);
    @(posedge clock); 
    #0.1

    // TODO why does this not work for the "cycles" buffering strategy?
    // wait for one cycle to ensure the data is transfered

    in0_valid = 1;
    inCtrl_valid = 1;
    in0_data = 24;

    #0.1

    wait(in0_ready == 1 && inCtrl_ready == 1);
    @(posedge clock); 
    #0.1

    in0_valid = 0;
    inCtrl_valid = 0;
  end

  logic [15:0] resCnt = 0;
  always @(posedge clock) begin
    if(clkCnt == 10000) begin
      $finish();
    end
    if(out0_valid == 1) begin
      $display("Result=%d", out0_data);
      resCnt += 1;
    end

    if(resCnt == 2) begin
      $finish();
    end
    clkCnt += 1;
  end
endmodule // driver
