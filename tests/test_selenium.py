from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def test_application():
    driver = webdriver.Chrome()

    try:
        driver.get("http://localhost:8000/docs")

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".swagger-ui")
            )
        )

        assert "Churn Prediction API" in driver.title

    finally:
        driver.quit()
