""" main.py

run the event study for a single stock.
"""
from event_study import download
from event_study import mk_rets
from event_study import mk_events
from event_study import mk_cars
from event_study import test_hypo


def main(tic, update_csv=True):
    """ Implements the event study for a given stock ticker `tic`.

    Parameters
    ----------
    tic : str
        Ticker

    update_csv : bool
        If True, data will be downloaded. Defaults to True.

    Notes
    -----


    """
    # Step 1: Download stock price and recommendation data for `tic`
    if update_csv is True:
        download.get_data(tic)
    else:
        print("Parameter `update_csv` set to False, skipping downloads...")

    # Step 2: Create a data frame with stock (tic) and market returns
    ret_df = mk_rets.mk_ret_df(tic)

    # Step 3: Create a data frame with the events
    event_df = mk_events.mk_event_df(tic)

    # Step 4: Calculate CARs for each event
    cars_df = mk_cars.mk_cars_df(ret_df, event_df)

    # Step 5: Hypothesis testing using t-statistics
    res = test_hypo.calc_tstats(cars_df)
    print(res)


if __name__ == "__main__":
    tic = 'TSLA'
    # NOTE: Keep update_csv = False because the yfinance API is broken
    update_csv = False
    main(tic=tic, update_csv=update_csv)