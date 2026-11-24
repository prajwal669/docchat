"""Generate the sample PDFs used by the demo, the tests and the evaluation.

All content is original and fictional. Each page of each document is a separate
topic, so page-level citation accuracy can be measured. Two documents (Northwind
and Fabrikam) deliberately cover the same topics with different numbers: a
retriever has to pick the right company, not just the right topic.

Usage:  python scripts/make_sample_docs.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DOCS: dict[str, list[tuple[str, str]]] = {
    "Northwind_Employee_Handbook": [
        ("Working Hours and Remote Work",
         "Northwind Labs core hours are 10:00 to 16:00 local time, Monday to Friday. Outside core hours, "
         "employees may arrange their schedules with their manager as long as they complete 40 hours per week. "
         "Employees may work remotely up to three days per week; the two remaining days are in-office anchor days, "
         "which are Tuesday and Thursday. Fully remote arrangements require written approval from a Director and "
         "are reviewed every six months. Remote employees receive a one-time home office stipend of $600 and a "
         "monthly internet allowance of $40. Overtime is not paid for exempt roles, but managers must ensure that "
         "workloads stay reasonable. Employees who work beyond 45 hours in a week for two consecutive weeks should "
         "notify their manager so that workload can be rebalanced."),
        ("Leave Policy",
         "Full-time employees accrue 24 days of paid annual leave per year, accrued monthly. Up to 8 unused days "
         "may be carried over into the next calendar year; any excess expires on March 31. Sick leave is separate: "
         "employees receive 10 paid sick days annually, and a doctor's note is required after three consecutive days "
         "of absence. Parental leave is 16 weeks at full pay for the primary caregiver and 6 weeks for the secondary "
         "caregiver, and it may be taken at any point within the first year after birth or adoption. Bereavement "
         "leave covers 5 days for immediate family and 2 days for extended family. Leave requests of more than five "
         "consecutive days should be submitted at least 30 days in advance through the HR portal."),
        ("Expenses and Travel",
         "Business expenses are reimbursed within 14 days of submission through the Expense Portal. Receipts are "
         "required for any expense over $25. Employees must book flights at least 14 days in advance and use economy "
         "class for flights shorter than six hours. The daily meal allowance during travel is $65, and hotel stays "
         "are capped at $220 per night in major cities and $150 elsewhere. Rental cars require approval from a "
         "manager and must be compact or mid-size unless more than three people are traveling together. Personal "
         "expenses and alcohol are not reimbursable. Expense reports submitted more than 60 days after the expense "
         "date will be rejected."),
        ("IT Security and Devices",
         "All company laptops must use full-disk encryption and automatic screen lock after five minutes of "
         "inactivity. Passwords must be at least 14 characters long and are stored in the company password manager; "
         "employees must enable multi-factor authentication using a hardware security key or authenticator app, and "
         "SMS codes are not accepted. Lost or stolen devices must be reported to the Security team within 2 hours "
         "using the hotline 555-0142. Personal devices may access company email only through the managed mobile app. "
         "Software installation requires approval from IT, and USB storage devices are blocked by default. Phishing "
         "emails should be forwarded to the phishing report mailbox rather than deleted."),
    ],
    "Fabrikam_Contractor_Guide": [
        ("Hours and Availability",
         "Fabrikam contractors set their own working hours but must be reachable between 12:00 and 15:00 local time "
         "on weekdays. Contractors are billed per hour and must submit timesheets every Friday by 17:00. Work on "
         "weekends requires prior written approval from the project lead and is billed at 1.5 times the normal rate. "
         "Contractors work remotely by default and are expected to attend on-site meetings at most once per month. "
         "A project kickoff call is scheduled within the first three business days of every engagement, and "
         "contractors must complete a short onboarding checklist before billing begins."),
        ("Time Off and Absences",
         "Fabrikam contractors do not receive paid annual leave, but they may take up to 15 unpaid days off per year "
         "with 14 days' notice to the project lead. Illness does not require a doctor's note, though absences beyond "
         "five consecutive working days must be reported to the engagement manager. Public holidays in the "
         "contractor's country of residence are not billed. If a contractor is unavailable for more than 10 "
         "consecutive working days, Fabrikam may assign a replacement. Contractors are asked to arrange a handover "
         "document whenever they are away for longer than three days."),
        ("Invoices and Expenses",
         "Invoices are issued monthly and paid within 45 days of receipt. Late invoices submitted after the 5th of "
         "the following month are paid in the next cycle. Travel expenses are reimbursed only if pre-approved in "
         "writing, and mileage is reimbursed at $0.55 per mile. Contractors must book their own accommodation within "
         "a nightly cap of $180 and submit receipts for all expenses above $50. Software licenses and equipment are "
         "the contractor's own responsibility unless stated in the contract. Fabrikam does not provide per diem meal "
         "allowances."),
        ("Security and Confidentiality",
         "Contractors must sign a non-disclosure agreement before accessing any Fabrikam systems. Access is granted "
         "through a virtual desktop, and downloading client data to personal machines is prohibited. Credentials must "
         "never be shared, and two-factor authentication is mandatory via an authenticator app. Security incidents "
         "must be reported to the engagement manager within 24 hours. Access is revoked automatically on the final "
         "day of the contract, and contractors must confirm deletion of any local project files within five business "
         "days of completion."),
    ],
    "Git_Essentials": [
        ("Getting Started",
         "Git is a distributed version control system that records snapshots of your project over time. Run git init "
         "to create a new repository in the current folder, or git clone followed by a URL to copy an existing one. "
         "The working directory holds your files, the staging area holds changes selected for the next snapshot, and "
         "the repository stores committed history. Use git status to see which files are modified or staged. Stage "
         "changes with git add and record them with git commit -m followed by a message. Good commit messages use the "
         "imperative mood and describe why a change was made. Use git log --oneline for a compact history of commits."),
        ("Branching and Merging",
         "A branch is a lightweight movable pointer to a commit. Create and switch to a new branch with git switch -c "
         "feature-login. The default branch is usually called main. To combine work, switch to the target branch and "
         "run git merge followed by the branch name. A fast-forward merge simply moves the pointer when no divergent "
         "commits exist, while a three-way merge creates a new merge commit. When two branches change the same lines, "
         "Git reports a merge conflict; open the marked files, choose the correct content, remove the conflict "
         "markers, then stage the files and commit. Delete a merged branch with git branch -d."),
        ("Undoing Changes",
         "To discard uncommitted edits in a file, use git restore followed by the file name. To unstage a file, use "
         "git restore --staged. If you made a mistake in the last commit message, git commit --amend lets you rewrite "
         "it. git reset moves the branch pointer backward: --soft keeps changes staged, --mixed (the default) keeps "
         "them unstaged, and --hard discards them completely, so use it carefully. For commits already shared with "
         "others, prefer git revert, which creates a new commit that undoes an earlier one without rewriting history. "
         "The reflog, shown by git reflog, lists recent positions of HEAD and can recover commits that seem lost."),
        ("Working with Remotes",
         "A remote is a named reference to another copy of the repository; the default remote is called origin. "
         "git fetch downloads new commits without changing your working files, whereas git pull fetches and then "
         "merges them. Publish local commits with git push, and use git push -u origin branch-name the first time to "
         "set the upstream. A pull request is a request on a hosting platform to review and merge a branch. git "
         "rebase replays your commits on top of another branch to produce a linear history, but you should never "
         "rebase commits that others have already pulled. Use .gitignore to keep build outputs and secrets out of "
         "the repository."),
    ],
    "Home_Gardening_Guide": [
        ("Soil and Compost",
         "Healthy vegetables start with healthy soil. Most garden plants prefer a slightly acidic to neutral pH "
         "between 6.0 and 7.0, which you can measure with an inexpensive test kit. Loamy soil, a balance of sand, "
         "silt, and clay, drains well while holding moisture. Add two to three inches of compost each spring to "
         "improve structure and supply nutrients. A good compost pile balances brown materials such as dry leaves and "
         "cardboard with green materials such as vegetable scraps and grass clippings, at roughly three parts brown "
         "to one part green. Turn the pile every one to two weeks, and it should be ready in two to three months. "
         "Avoid adding meat, dairy, or pet waste."),
        ("Watering and Mulching",
         "Most vegetable gardens need about one inch of water per week, delivered in one or two deep soakings rather "
         "than daily sprinkles. Deep watering encourages roots to grow downward and makes plants more resilient in "
         "dry spells. Water in the early morning so leaves dry quickly, which reduces fungal problems. Drip "
         "irrigation and soaker hoses deliver water directly to the roots and waste less than overhead sprinklers. "
         "Spread a two to three inch layer of mulch, such as straw or shredded bark, around plants to retain "
         "moisture, suppress weeds, and keep soil temperatures even. Keep mulch a few inches away from stems to "
         "prevent rot."),
        ("Pests and Diseases",
         "Check plants weekly for signs of trouble, since early detection makes control easier. Aphids, small "
         "soft-bodied insects clustered on new growth, can be knocked off with a strong spray of water or treated "
         "with insecticidal soap. Slugs feed at night and can be trapped with shallow dishes of beer or deterred with "
         "copper tape. Powdery mildew appears as a white coating on leaves and is best prevented by spacing plants "
         "for airflow. Blight on tomatoes causes dark spots and wilting; remove affected plants and do not compost "
         "them. Rotate crop families each year to break pest and disease cycles. Attract beneficial insects such as "
         "ladybugs by planting flowers like marigolds and alyssum."),
        ("Seasonal Planting Calendar",
         "In a temperate climate, plant cool-season crops such as peas, spinach, and lettuce in early spring, as soon "
         "as the soil can be worked. Warm-season crops such as tomatoes, peppers, and squash should wait until after "
         "the last frost, when nighttime temperatures stay above 10 degrees Celsius. Start tomato and pepper seeds "
         "indoors six to eight weeks before the last frost date. In midsummer, sow a second round of carrots and "
         "beets for autumn harvest. Garlic is planted in October for harvest the following July. Finish the year by "
         "clearing spent plants and adding a layer of compost or cover crop such as clover to protect the bare soil "
         "over winter."),
    ],
    "Personal_Finance_Basics": [
        ("Budgeting",
         "A budget is a plan for how you will spend your income each month. The 50/30/20 rule suggests allocating 50 "
         "percent of after-tax income to needs such as rent and groceries, 30 percent to wants such as dining out and "
         "entertainment, and 20 percent to saving and paying down debt. Start by tracking every expense for one month "
         "to see where money actually goes. Zero-based budgeting assigns every dollar a job so that income minus "
         "planned spending equals zero. Review your budget monthly and adjust categories when circumstances change. "
         "Automating bill payments helps avoid late fees."),
        ("Emergency Fund and Saving",
         "An emergency fund is cash set aside for unexpected costs such as medical bills or job loss. Most advisers "
         "recommend saving three to six months of essential living expenses, and people with irregular income may aim "
         "for closer to twelve months. Keep the fund in a high-yield savings account, where it earns interest and "
         "remains easy to access, rather than in stocks. Build it gradually by automatically transferring a fixed "
         "amount on each payday. Start with a small first goal, such as 1,000 dollars, then work toward the full "
         "target. Only use the fund for true emergencies, and replenish it afterwards."),
        ("Managing Debt",
         "The debt avalanche method pays minimum amounts on all debts and puts extra money toward the debt with the "
         "highest interest rate, which minimizes total interest paid. The debt snowball method instead targets the "
         "smallest balance first, which gives quick wins and motivation even though it may cost more in interest. "
         "Credit card interest rates commonly exceed 20 percent, so paying these cards off is often a priority. "
         "Always pay at least the minimum on time, since late payments harm your credit score. Consolidating several "
         "debts into one loan with a lower interest rate can simplify payments, but check for fees. Avoid taking new "
         "debt while paying off old balances."),
        ("Investing Basics",
         "Investing allows money to grow faster than inflation over long periods. An index fund holds a broad basket "
         "of stocks or bonds and tracks a market index, offering low fees and instant diversification. "
         "Diversification, spreading money across many assets, reduces the risk that one failing company harms your "
         "portfolio. Compound interest means earnings generate their own earnings; at a 7 percent annual return, "
         "money roughly doubles every ten years, following the rule of 72. Time in the market usually matters more "
         "than timing the market. Younger investors can often hold a higher share of stocks, while those nearing "
         "retirement shift toward bonds. Expense ratios above 1 percent can noticeably erode long-term returns."),
    ],
}


def build(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], fontSize=18, spaceAfter=14)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=11, leading=16)
    paths = []
    for name, pages in DOCS.items():
        path = out_dir / f"{name}.pdf"
        story = []
        for i, (heading, text) in enumerate(pages):
            story += [Paragraph(heading, title), Spacer(1, 6), Paragraph(text, body)]
            if i < len(pages) - 1:
                story.append(PageBreak())  # exactly one topic per page
        SimpleDocTemplate(
            str(path), pagesize=A4, title=name.replace("_", " "), author="DocChat sample data"
        ).build(story)
        paths.append(path)
    return paths


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    for p in build(target):
        print("wrote", p)
