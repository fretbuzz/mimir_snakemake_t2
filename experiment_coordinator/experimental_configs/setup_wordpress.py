# doesn't do much ATM, but in like 2 hours I could automate the whole setup for wordpress...

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import sys
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

def install_pluggin():
    try:
        faker_press_install = driver.find_element_by_id('plugin_install_from_iframe')
        print "faker_press_install", faker_press_install
        # faker_press_install = driver.find_element_by_class_name('install-now button')
        # for elem in login_form:
        #    print 'element.text: {0}'.format(elem)
        print faker_press_install
        faker_press_install.click()

        class_of_activate_button = "button-primary"
        faker_press_activate = driver.find_element_by_class_name(class_of_activate_button)
        print "faker_press_activate", faker_press_activate
        faker_press_activate.click()
    except:
        print("there was an exception thrown in installing/activating pluggin...")
        pass  # might already be installed...

if __name__== "__main__":
    if len(sys.argv) <= 2:
        print "needs an ip address, budy"

    print sys.argv
    ip_of_wp = sys.argv[1]
    port_of_wp = sys.argv[2]

    #firefox_binary=FirefoxBinary('/anaconda2/lib/python2.7/site-packages/selenium/webdriver/firefox')
    driver = webdriver.Firefox()
    #driver.get("http://www.python.org")
    driver.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin')
    #assert "Python" in driver.title
    #elem = driver.find_element_by_name("q")
    elem = driver.find_element_by_name("log")
    elem.clear()
    elem.send_keys("user")
    elem = driver.find_element_by_name("pwd")
    elem.clear()
    elem.send_keys("HSjDUn1Vlt")
    elem.send_keys(Keys.RETURN)
    print elem
    time.sleep(5)
    #driver.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=search&s=faker')

    # okay, first install fakerpress
    '''
    page_about_fakerpress = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=fakerpress&TB_iframe=true&height=-34%22&width=772'
    driver.get(page_about_fakerpress)
    install_pluggin()
    time.sleep(10)

    # then install "Export All URLs"
    page_about_export_all_urls = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=export-all-urls&TB_iframe=true&width=772&height=627'
    driver.get(page_about_export_all_urls)
    install_pluggin()
    time.sleep(10)

    # then install "app_pass"
    page_about_app_pass = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=application-passwords&TB_iframe=true&width=772&height=627'
    driver.get(page_about_app_pass)
    install_pluggin()
    '''

    # now generate the fake data using fakerpress
    max_num = 'fakerpress-field-qty-max'
    max_num_class = 'fp-field fp-type-number fp-size-tiny'
    min_num = 'fakerpress-field-qty-min'
    drop_down_id = 's2id_fakerpress-field-meta-type'

    user_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=users'
    driver.get(user_page)
    roles_field_class = 'select2-input'#''select2-choices'
    rolesBox = driver.find_element_by_class_name(roles_field_class)
    rolesBox.send_keys('Administrator')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Editor')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Author')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Contributor')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Subscriber')
    rolesBox.send_keys(Keys.RETURN)

    #max_quantity_box = driver.find_element_by_id(max_num)
    max_quantity_box = driver.find_element_by_id(max_num)
    max_quantity_box.click()
    max_quantity_box.send_keys("200")
    min_quantity_box = driver.find_element_by_id(min_num)
    min_quantity_box.send_keys("200")
    time.sleep(2)

    terms_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=terms'
    driver.get(terms_page)
    time.sleep(2)
    comments_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=comments'
    driver.get(comments_page)
    time.sleep(2)
    post_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=posts'
    driver.get(post_page)
    time.sleep(2)

    dropDownBox = Select(driver.find_element_by_id(drop_down_id))
    dropDownBox.select_by_visible_text('Banana')


    '''
    IWebElement dropDownListBox = driver.findElement(By.Id("selection"));
    SelectElement clickThis = new SelectElement(dropDownListBox);
    clickThis.SelectByText("Germany");
    '''

    #elem.send_keys("pycon")
    #elem.send_keys(Keys.RETURN)
    #assert "No results found." not in driver.page_source
    time.sleep(300)
    driver.close()
