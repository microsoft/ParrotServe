def write_synopsis(title):
    """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

    Title: {title}
    Playwright: This is a synopsis for the above play:"""


def write_review(synopsis):
    """You are a play critic from the New York Times.
    Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""


def write_post(time, location, synopsis, review):
    """You are a social media manager for a theater company.
    Given the title of play, the era it is set in, the date,time and location, the synopsis of the play, and the review of the play, it is your job to write a social media post for that play.

    Here is some context about the time and location of the play:
    Date and Time: {time}
    Location: {location}

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:
    {review}

    Social Media Post:
    """


def main():
    play_infos = [...]
    for title, time, location in play_infos:
        synopsis = write_synopsis(title)
        review = write_review(synopsis)
        post = write_post(time, location, synopsis, review)
        print(post)
