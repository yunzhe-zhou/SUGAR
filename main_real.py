from SUGAR.real_infer import *
import warnings
warnings.filterwarnings("ignore")

# =================================================================================================================
# Main function to implements the real data analysis in section 6 of the paper
# =================================================================================================================

def main():
    # consider different performance group
    run_real("low")
    run_real("high")
    
    print_real("low")
    print_real("high")
    
if __name__ == '__main__':
    main()