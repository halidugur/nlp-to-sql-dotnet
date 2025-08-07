using Microsoft.AspNetCore.Mvc;
using stajprojem.Data;
using stajprojem.Models;

public class OrderDetailsController : Controller
{
    private readonly AppDbContext _context;
    public OrderDetailsController(AppDbContext context)
    {
        _context = context;
    }

    public IActionResult Index()
    {
        var orderDetails = _context.OrderDetails.ToList();
        return View(orderDetails);
    }
}
