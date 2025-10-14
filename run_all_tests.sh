#!/bin/bash
# Run all tests for Advanced Statistical Analysis Toolkit
# Author: Gabriel Demetrios Lafis

echo "======================================================================"
echo "Advanced Statistical Analysis Toolkit - Test Suite"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
PYTHON_TESTS_PASSED=0
R_TESTS_PASSED=0

# Check if Python is available
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python3 found${NC}"
    PYTHON_VERSION=$(python3 --version)
    echo "  $PYTHON_VERSION"
else
    echo -e "${RED}✗ Python3 not found${NC}"
    echo "  Please install Python 3.8 or higher"
fi

# Check if R is available
echo ""
echo "Checking R installation..."
if command -v Rscript &> /dev/null; then
    echo -e "${GREEN}✓ R found${NC}"
    R_VERSION=$(Rscript --version 2>&1)
    echo "  $R_VERSION"
else
    echo -e "${YELLOW}⚠ R not found${NC}"
    echo "  R tests will be skipped"
    echo "  Please install R to run R tests"
fi

echo ""
echo "======================================================================"
echo "Running Python Tests"
echo "======================================================================"
echo ""

# Check if pytest is installed
if python3 -c "import pytest" 2>/dev/null; then
    echo "Running pytest..."
    cd tests
    if python3 -m pytest test_python_integration.py -v --tb=short; then
        echo -e "\n${GREEN}✓ Python tests PASSED${NC}"
        PYTHON_TESTS_PASSED=1
    else
        echo -e "\n${RED}✗ Python tests FAILED${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠ pytest not installed${NC}"
    echo "  Installing pytest..."
    pip3 install pytest
    
    echo "Running pytest..."
    cd tests
    if python3 -m pytest test_python_integration.py -v --tb=short; then
        echo -e "\n${GREEN}✓ Python tests PASSED${NC}"
        PYTHON_TESTS_PASSED=1
    else
        echo -e "\n${RED}✗ Python tests FAILED${NC}"
    fi
    cd ..
fi

echo ""
echo "======================================================================"
echo "Running R Tests"
echo "======================================================================"
echo ""

if command -v Rscript &> /dev/null; then
    echo "Running R unit tests..."
    cd tests
    if Rscript test_r_functions.R; then
        echo -e "\n${GREEN}✓ R tests PASSED${NC}"
        R_TESTS_PASSED=1
    else
        echo -e "\n${RED}✗ R tests FAILED${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠ Skipping R tests (R not installed)${NC}"
fi

echo ""
echo "======================================================================"
echo "Test Summary"
echo "======================================================================"
echo ""

if [ $PYTHON_TESTS_PASSED -eq 1 ]; then
    echo -e "${GREEN}✓ Python tests: PASSED${NC}"
else
    echo -e "${RED}✗ Python tests: FAILED${NC}"
fi

if command -v Rscript &> /dev/null; then
    if [ $R_TESTS_PASSED -eq 1 ]; then
        echo -e "${GREEN}✓ R tests: PASSED${NC}"
    else
        echo -e "${RED}✗ R tests: FAILED${NC}"
    fi
else
    echo -e "${YELLOW}⚠ R tests: SKIPPED (R not installed)${NC}"
fi

echo ""

# Overall result
if [ $PYTHON_TESTS_PASSED -eq 1 ] && ([ $R_TESTS_PASSED -eq 1 ] || ! command -v Rscript &> /dev/null); then
    echo -e "${GREEN}======================================================================"
    echo "All available tests PASSED!"
    echo "======================================================================${NC}"
    exit 0
else
    echo -e "${RED}======================================================================"
    echo "Some tests FAILED!"
    echo "======================================================================${NC}"
    exit 1
fi
