import javax.swing.JFrame;

public class VentanaEmpresa extends JFrame{

	public VentanaEmpresa(){
		super("Ventana Empresa");
		
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setLocation(400, 100);
		PanelEmpresa p1=new PanelEmpresa();
		this.add(p1);
		this.pack();
		this.setVisible(true);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
